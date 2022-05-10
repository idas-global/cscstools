import os
import json
import random
import hashlib
import datetime
import itertools
import pandas as pd
from tqdm import tqdm
from noteclasses import ImageBMP, ImageHyperSpec, ImageLP, Spectrum, Side, FileType
from pathlib import Path


class NoteManager:
    def __init__(self, dbPath, index=True):
        if not os.path.exists(dbPath):
            print(f'The specified path {dbPath} does not exist')
            raise FileNotFoundError

        self.path = dbPath
        self.data = None

        if index:
            self.index()

    def index(self, validate=True):
        print('Discovering notes...')
        self.fileList = []
        self.dirList  = []
        for path, subdirs, files in os.walk(self.path):
            self.dirList += [f'{path}/{d}' for d in subdirs]
            self.fileList += [f'{path}/{f}' for f in files]

        dirShortList = [d for d in self.dirList if not checkIfDirectoryContainsADirectory(d)]
        shortDict    = {d : os.listdir(d) for d in dirShortList}

        if validate:
            self.validate(shortDict)

        print(f'Found {len(shortDict)} notes in database at {self.path}')

        self.validNoteList = list(shortDict.keys())
        self.validNoteList.sort()
        self.data          = pd.DataFrame([loadNoteMeta(n) for n in self.validNoteList])


    def validate(self, shortDict):
        print('Checking validity of notes...')
        invalidList = []
        for notedir, files in tqdm(shortDict.items()):
            valid, missing = checkIfNoteDirIsValid(notedir, files, self.path)
            if not valid:
                missingFormat = [f'{m[0].value}-{m[1].value}' for m in missing]
                print(f'{notedir} MISSES {len(missing)}: {missingFormat}')
                invalidList.append(notedir)

        for inv in invalidList:
            del shortDict[inv]

    def filter(self, phrase=None, keywords=None):
        df = self.data.copy()
        if isinstance(phrase, str):
           df = filterByColumn(df, 'phrase', phrase)
        if isinstance(phrase, list):
           for p in phrase:
               df = filterByColumn(df, 'phrase', p)

        if isinstance(keywords, str):
           df = filterByColumn(df, 'phrase', phrase)
        if isinstance(keywords, list):
           for k in keywords:
               df = filterByColumn(df, 'keywords', k)
        self.data = df
        return self


class Note:
    def __init__(self, dic):
        self.path = dic['path']
        self.noteid = self.path.split('/')[-1]
        self.keywords = dic['keywords']
        self.phrase = dic['phrase']
        self.data = dic

    def load(self, spectrum, side, straighten=False):
        keyString = f'{spectrum.value}-{side.value}'
        if not self.data['spectrum-sides'][keyString]:
            print(f'The requested spectrum-side {keyString} does not exist in {self.path}')
            raise KeyError

        if spectrum in [Spectrum.RGB, Spectrum.INFRARED]:
            fPath = f'{self.path}/{self.noteid}_{spectrum.value}_{side.value}.bmp'
            return ImageBMP(fPath, rotation=90, straighten=straighten)

        if spectrum == Spectrum.HYPERSPEC:
            hdrPath = f'{self.path}/{self.noteid}_{spectrum.value}_{side.value}.hdr'
            rawPath = f'{self.path}/{self.noteid}_{spectrum.value}_{side.value}.raw'
            return ImageHyperSpec(hdrPath, rawPath, straighten=straighten)

        if spectrum == Spectrum.LASERPROFILE:
            fPath = f'{self.path}/{self.noteid}_{spectrum.value}_{side.value}.npy'
            im = ImageLP(fPath, preprocess=True)
            if straighten:
                im.flatten()
            return im.projectTo2D(straighten=False, normalise=True)


def filterByColumn(df, name, filterVal):
    if isinstance(filterVal, list):
        df = df[df[name].isin(filterVal)]
    if isinstance(filterVal, str):
        df = df[df[name] == filterVal]
    return df


def loadNoteMeta(noteDir):
    metaPath = f'{noteDir}/meta.json'
    if not os.path.exists(metaPath):
        print(f'No meta file found in {noteDir}. Consider running NoteManager.index(validate=True)')
    with open(metaPath, 'r') as rf:
        return json.load(rf)


def checkIfNoteDirIsValid(notedir, files, basePath):
    valid, missing = checkIfNoteFilesArePresent(notedir, files)
    updateNoteMeta(notedir, basePath, missing)
    return valid, missing


def updateNoteMeta(notedir, basePath, missing):
    metaFile = f'{notedir}/meta.json'
    updateFlag = False
    if not os.path.exists(metaFile):
        updateFlag = True
        with open(metaFile, 'w+') as wf:
            json.dump({}, wf)
    try:
        with open(metaFile, 'r') as rf:
            metaDict = json.load(rf)
    except:
        metaDict = {}

    if 'time-entered' not in metaDict:
        metaDict['time-entered'] = datetime.datetime.now().isoformat()
        updateFlag = True

    if 'keywords' not in metaDict:
        metaDict['keywords'] = notedir.replace(basePath, '').split('/')[1:]

    if 'path' not in metaDict:
        updateFlag = True
        metaDict['path'] = notedir

    if 'phrase' not in metaDict:
        metaDict['phrase'] = getIdentityPhrase()
        updateFlag = True

    # Identifiers
    key = 'identifiers'
    if 'identifiers' not in metaDict:
        metaDict[key] = {}
    identifiers = ['serial-num', 'series', 'front-plate', 'back-plate', 'fed-indicator']
    for identifier in identifiers:
        if identifier not in metaDict[key]:
            metaDict[key][identifier] = None

    # 16-04 Data
    key = '16-04-data'
    if key not in metaDict:
        metaDict[key] = {}
    data = ['time-handed-in', 'location']
    for d in data:
        if d not in metaDict[key]:
            metaDict[key][d] = None

    # USSS Labels
    key = 'usss-labels'
    if key not in metaDict:
        metaDict[key] = {}
    data = ['circular', 'parent']
    for d in data:
        if d not in metaDict[key]:
            metaDict[key][d] = None

    # Side spectrum presence check
    key = 'spectrum-sides'
    if key not in metaDict:
        metaDict[key] = {}
    for spectrum, side in itertools.product(Spectrum, Side):
        if (spectrum, side) in missing:
            metaDict[key][f'{spectrum.value}-{side.value}'] = False
        else:
            metaDict[key][f'{spectrum.value}-{side.value}'] = True

    if updateFlag:
        metaDict['last-updated'] = datetime.datetime.now().isoformat()

    with open(metaFile, 'w+') as wf:
        json.dump(metaDict, wf)


def checkIfNoteFilesArePresent(notedir, files):
    noteid = notedir.split('/')[-1]

    expectedFiles = [
        # Check RGB
        ((Spectrum.RGB, Side.FRONT), f'{noteid}_{Spectrum.RGB.value}_{Side.FRONT.value}.{FileType.BITMAP.value}'),
        ((Spectrum.RGB, Side.BACK), f'{noteid}_{Spectrum.RGB.value}_{Side.BACK.value}.{FileType.BITMAP.value}'),

        # Check NIR
        ((Spectrum.INFRARED, Side.FRONT), f'{noteid}_{Spectrum.INFRARED.value}_{Side.FRONT.value}.{FileType.BITMAP.value}'),
        ((Spectrum.INFRARED, Side.BACK), f'{noteid}_{Spectrum.INFRARED.value}_{Side.BACK.value}.{FileType.BITMAP.value}'),

        # Check HyperSpec
        ((Spectrum.HYPERSPEC, Side.FRONT), f'{noteid}_{Spectrum.HYPERSPEC.value}_{Side.FRONT.value}.{FileType.HDR.value}'),
        ((Spectrum.HYPERSPEC, Side.BACK), f'{noteid}_{Spectrum.HYPERSPEC.value}_{Side.FRONT.value}.{FileType.RAW.value}'),
        ((Spectrum.HYPERSPEC, Side.FRONT), f'{noteid}_{Spectrum.HYPERSPEC.value}_{Side.BACK.value}.{FileType.HDR.value}'),
        ((Spectrum.HYPERSPEC, Side.BACK), f'{noteid}_{Spectrum.HYPERSPEC.value}_{Side.BACK.value}.{FileType.RAW.value}'),

        # Check Laser
        ((Spectrum.LASERPROFILE, Side.FRONT), f'{noteid}_{Spectrum.LASERPROFILE.value}_{Side.FRONT.value}.{FileType.NUMPY.value}'),
        ((Spectrum.LASERPROFILE, Side.BACK), f'{noteid}_{Spectrum.LASERPROFILE.value}_{Side.BACK.value}.{FileType.NUMPY.value}')
        ]

    missingFiles = [chk[0] for chk in expectedFiles if chk[1] not in files]

    if len(missingFiles) == len(expectedFiles):
        sideIndexFiles = [(chk[0], chk[1].replace(Side.FRONT.value, '0').replace(Side.BACK.value, '1')) for chk in expectedFiles]
        missingFiles = [chk[0] for chk in sideIndexFiles if chk[1] not in files]

    return not bool(missingFiles), missingFiles


def getIdentityPhrase():
    a = ['diminuitive', 'tiny', 'small', 'average', 'medium', 'large', 'big', 'huge', 'enormous', 'massive', 'giant', 'baby']
    b = ['red', 'purple', 'green', 'blue', 'grey', 'yellow', 'brown', 'teal', 'cyan', 'magenta', 'orange', 'aquamarine', 'indigo']
    c = ['awkward', 'brave', 'simple', 'reasonable', 'angry', 'funny', 'irritated', 'sad', 'happy', 'crazy', 'sleepy', 'silly', 'sleeping', 'snoozing']
    d = ['aardvark', 'bird', 'cat', 'dog', 'elephant', 'flamingo', 'giraffe', 'hippo', 'iguana', 'kangaroo', 'llama', 'mouse']

    return [random.choice(a), random.choice(b), random.choice(c), random.choice(d)]


def checkIfDirectoryContainsADirectory(path):
    return any([os.path.isdir(f'{path}/{p}') for p in os.listdir(path)])


def getDirectoryHash(directory):
    hash = hashlib.md5()
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            if 'meta.json' in f:
                continue
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash.update(chunk)
    return hash.hexdigest()


def main():
    notedb = NoteManager('/mnt/data/Packs')


if __name__ == '__main__':
    main()
