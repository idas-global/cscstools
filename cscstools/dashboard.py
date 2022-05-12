import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image

import dash
from dash import html, dcc, callback_context, dash_table
from dash.dependencies import Input, Output, State

from noteclasses import Spectrum, Side
from notemanager import Note

import plotly.graph_objects as go


def getAdjustedROI(arr, scaleSize, roi):
    yMin = int(arr.shape[0] * roi['yMin'] / scaleSize[0])
    yMax = int(arr.shape[0] * roi['yMax'] / scaleSize[0])
    xMin = int(arr.shape[1] * roi['xMin'] / scaleSize[1])
    xMax = int(arr.shape[1] * roi['xMax'] / scaleSize[1])
    return {'xMin': xMin, 'xMax': xMax, 'yMin': arr.shape[0] - yMax, 'yMax': arr.shape[0] - yMin}


def generatePlot(note, spectrum, side, crop=None, straighten=False):
    imgWidth = 1200
    imgHeight = 500

    imObj = note.load(spectrum, side, straighten=straighten)
    if spectrum in [Spectrum.RGB, Spectrum.INFRARED]:
        im = imObj.array

    if spectrum == Spectrum.HYPERSPEC:
        im, wavelength = imObj.getBand(120)
        im = (im*255).astype(np.uint8)

    if spectrum == Spectrum.LASERPROFILE:
        im = (imObj*255).astype(np.uint8)

    if im.shape[0] > im.shape[1]:
        im = cv2.rotate(imObj.array, 0)

    if crop:
        roi = getAdjustedROI(im, (imgHeight, imgWidth), crop)
        if len(im.shape) == 2:
            im = im[roi['yMin']:roi['yMax'], roi['xMin']:roi['xMax']]
        else:
            im = im[roi['yMin']:roi['yMax'], roi['xMin']:roi['xMax'], :]

    im = cv2.resize(im, (imgWidth, imgHeight))
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(im)
    # Create figure
    fig = go.Figure()

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, imgWidth],
            y=[0, imgHeight],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, imgWidth]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, imgHeight],
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=imgWidth,
            y=imgHeight,
            sizey=imgHeight,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=im)
    )

    # Configure other layout
    fig.update_layout(
        width=imgWidth,
        height=imgHeight,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig


def formatJSONForDisplay(jDict, subdict=False):
    outputStr = 'META DATA\n\n' if not subdict else '\n\n'
    for key, val in jDict.items():
        if isinstance(val, list):
            val = ' '.join(val)
        if isinstance(val, dict):
            val = formatJSONForDisplay(val, subdict=True)
            key = key.upper()
        if subdict:
            outputStr += '\t' + str(key).ljust(14) + ' : ' + str(val) + '\n'
        else:
            outputStr += str(key).ljust(14) + ' : ' + str(val) + '\n'

    return outputStr.replace('\n\n', '\n')


def sanitiseJSONString(s):
    if s == 'True':
        s = True
    if s == 'False':
        s = False
    if s == 'None':
        s = None
    return s


def formatDisplayTextToJSON(metaData):
    jDict = {}
    splitList = metaData.split('\n')[1:]
    splitList = [r.split(' : ') for r in splitList if r != '']
    splitList = [(k.replace(' ', ''), v.split(' ') if ' ' in v else v) for k, v in splitList]

    for idx, (k, v) in enumerate(splitList):
        if '\t' in k:
            continue
        if v:
            jDict[k] = sanitiseJSONString(v)
            continue
        subList = []
        hop = 1
        while '\t' in splitList[idx+hop][0]:
            subList.append(splitList[idx+hop])
            hop += 1

        jDict[k.lower()] = {subk.replace('\t', '') : sanitiseJSONString(subv) for subk, subv in subList}

    return jDict


def formatDataFrame(df):
    dfDisplay = df[['path', 'phrase', 'keywords', 'time-entered']].copy()
    dfDisplay['keywords'] = df['keywords'].apply(lambda s: ' '.join(s))
    dfDisplay['phrase']   = df['phrase'].apply(lambda s: ' '.join(s))
    return dfDisplay


def dashboard(notedb):
    df = notedb.data
    app = dash.Dash(suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.Div([
            html.H5('NOTE VIEWER', style={'font-size': '40px', 'margin-top' : '0', 'margin-bottom' : '0'}),
            html.Button('PREVIOUS', n_clicks=0,
                        id='note-selection-previous',
                        style={'width': '5%', 'display': 'inline-block'}),
            html.Button('NEXT', n_clicks=0,
                        id='note-selection-next',
                        style={'width': '5%', 'display': 'inline-block'}),
            dcc.Dropdown(
                df['path'].unique(), df.loc[0, 'path'],
                id='note-selection',
                style={'width': '40%'}),
            dcc.RadioItems(
                [s.value for s in Spectrum], Spectrum.RGB.value,
                id='spectrum-selection',
                style={'width': '30%', 'display': 'inline-block'}),
            dcc.RadioItems(
                [s.value for s in Side], Side.FRONT.value,
                id='side-selection',
                style={'width': '25%', 'display': 'inline-block'}),
            dcc.RadioItems(
                ['Raw', 'Processed'], 'Raw',
                id='straighten-selection',
                style={'width': '14%', 'display': 'inline-block'})
            ]),
        html.Div([
            dcc.Graph(
                figure=generatePlot(Note(df.loc[0]), Spectrum.RGB, Side.FRONT),
                id='note-window',
                style={'width': '69%', 'display': 'inline-block'}),
            html.Div([
                dcc.Tabs(id='tabs-right-window',
                         value='meta-data',
                         children=[dcc.Tab(label='Meta Data', value='meta-data'),
                                   dcc.Tab(label='Features',  value='features')]),
                html.Div(id='tabs-right-window-content')], style={'width': '30%', 'display': 'inline-block'})
            ]),
        html.Div([
        dash_table.DataTable(formatDataFrame(df).to_dict('records'), [{"name": i, "id": i} for i in formatDataFrame(df).columns],
                             id='note-table',
                             style_table={'overflowY': 'scroll', 'height': 500})
            ])
    ])
    # TODO: ADD UPDATE OF TAB WHEN SWITCHING BETWEEN
    @app.callback(Output('tabs-right-window-content', 'children'),
                  Input('tabs-right-window', 'value'),
                  State('note-selection', 'value'))
    def render_content(tab, noteSelection):
        idx = df.index[df['path'] == noteSelection][0]
        dic = dict(df.loc[idx])
        if tab == 'meta-data':
            return [html.Button('SAVE', n_clicks=0,
                                id='save-note-metadata',
                                style={'width': '15%'}),
                    html.Button('RESET', n_clicks=0,
                                id='reset-note-metadata',
                                style={'width': '15%'}),
                    dcc.Textarea(
                        id='note-metadata',
                        value=formatJSONForDisplay(dic),
                        spellCheck=False,
                        style={'width': '100%', 'height': 450},
                        )
                    ]
        if tab == 'features':
            return [dcc.Textarea(
                        id='feature-data',
                        value='WAKKA WAKKA',
                        spellCheck=False,
                        style={'width': '100%', 'height': 450},
                        )
                    ]


    @app.callback(
        Output('note-window', 'figure'),
        Input('note-selection', 'value'),
        Input('spectrum-selection', 'value'),
        Input('side-selection', 'value'),
        Input('straighten-selection', 'value'),
        Input('note-window', 'relayoutData'))
    def selectNote(noteSelection, spectrumSelection, sideSelection, straightenSelection, relayoutData):
        try:
            cropROI = {'xMin' : relayoutData["xaxis.range[0]"],
                       'xMax' : relayoutData["xaxis.range[1]"],
                       'yMin' : relayoutData["yaxis.range[0]"],
                       'yMax' : relayoutData["yaxis.range[1]"]}
        except:
            cropROI = None
        idx = df.index[df['path'] == noteSelection][0]
        note = Note(df.loc[idx])
        straighten = True
        if straightenSelection == 'Raw':
            straighten = False
        return generatePlot(note, Spectrum(spectrumSelection), Side(sideSelection), crop=cropROI, straighten=straighten)

    @app.callback(
        Output('feature-data', 'value'),
        Input('note-window', 'relayoutData'))
    def getZoomSelection(data):
        try:
            xMin = data["xaxis.range[0]"]
            xMax = data["xaxis.range[1]"]
            yMin = data["yaxis.range[0]"]
            yMax = data["yaxis.range[1]"]
        except:
            return None
        return f'{json.dumps(data)}\n({yMin}, {xMin}), ({yMax}, {xMax})'

    @app.callback(
        Output('note-metadata', 'value'),
        Input('note-selection', 'value'),
        Input('save-note-metadata', 'n_clicks'),
        Input('reset-note-metadata', 'n_clicks'),
        State('note-metadata', 'value'))
    def displayMeta(noteSelection, saveButton, resetButton, metaData):
        changed = [p['prop_id'] for p in callback_context.triggered][0]
        idx = df.index[df['path'] == noteSelection][0]
        dic = dict(df.loc[idx])
        if 'note-selection' in changed:
            return formatJSONForDisplay(dic)
        if 'reset-note-metadata' in changed:
            return formatJSONForDisplay(dic)
        if 'save-note-metadata' in changed:
            try:
                jDict = formatDisplayTextToJSON(metaData)
            except:
                return 'ERROR: INCORRECT FORMAT FOR DATA\n\n' + formatJSONForDisplay(dic)
            with open(f'{noteSelection}/meta.json', 'w+') as wf:
                json.dump(jDict, wf)
            df.loc[idx] = pd.Series(jDict)
            return 'SAVED DATA\n\n' + formatJSONForDisplay(dic)

    @app.callback(
        Output('note-selection', 'value'),
        Input('note-selection-next', 'n_clicks'),
        Input('note-selection-previous', 'n_clicks'),
        Input('note-table', 'active_cell'),
        State('note-selection', 'value'))
    def cycleNote(nextClick, prevClick, tableCell, noteSelection):
        changed = [p['prop_id'] for p in callback_context.triggered][0]
        idx = df.index[df['path'] == noteSelection][0]
        if 'note-selection-next' in changed:
            if idx == len(df) - 1:
                idx = 0
            else:
                idx += 1
        if 'note-selection-previous' in changed:
            if idx == 0:
                idx = len(df) - 1
            else:
                idx -= 1
        if 'note-table' in changed:
            idx = tableCell['row']

        return df.loc[idx, 'path']

    app.run_server(debug=True, port=8050, host='0.0.0.0')
