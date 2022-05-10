from notemanager import NoteManager
from dashboard import dashboard

def main():
    notedb = NoteManager('/mnt/data/Packs')
    dashboard(notedb)

if __name__ == '__main__':
    main()