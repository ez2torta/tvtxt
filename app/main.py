from fasthtml.common import *
import json
import os
import redis
from dotenv import load_dotenv
load_dotenv()


redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST'),
    port=int(os.environ.get('REDIS_PORT', 6379)),
    username=os.environ.get('REDIS_USERNAME'),
    password=os.environ.get('REDIS_PASSWORD'),
    decode_responses=True
)

REDIS_KEY = 'tvtxt:latest'

def load_last_from_redis():
    data = redis_client.get(REDIS_KEY)
    if data:
        try:
            return json.loads(data)
        except Exception:
            return {'transcription': '', 'scene': '', 'action': ''}
    return {'transcription': '', 'scene': '', 'action': ''}

app,rt = fast_app(
    pico=False,
    hdrs = [
        Style("""
            body { background: #fff; margin: 0; padding: 0; }
            .script-page {
                font-family: 'Courier New', Courier, monospace;
                font-size: 12pt;
                background: #fff;
                margin: 32px auto;
                padding: 48px 60px 48px 90px;
                box-shadow: 0 0 0 1px #eee;
                max-width: 650px;
                min-height: 450px; /* Reducido a la mitad */
                position: relative;
            }
            .scene-heading { margin-top: 24px; margin-bottom: 8px; font-weight: bold; letter-spacing: 1px; }
            .action { margin-bottom: 20px; white-space: pre-line; }
            .character { text-transform: uppercase; margin-left: 170px; margin-top: 20px; margin-bottom: 0; font-weight: bold; }
            .dialogue { margin-left: 140px; margin-bottom: 12px; width: 330px; white-space: pre-line; }
            .page-number { position: absolute; top: 32px; right: 64px; font-size: 10pt; color: #bbb; }
            .disclaimer { margin-top: 40px; color: #888; font-size: 10pt; text-align: center; }
        """)
    ]
)

@rt("/")
def script_page(req):
    item = load_last_from_redis()
    return Div(
        Safe("<div class='page-number'>1</div>"),
        H4(item.get('scene', ''), _class="scene-heading"),
        Div(item.get('action', ''), _class="action"),
        Div("NARRATOR", _class="character"),
        Div(item.get('transcription', ''), _class="dialogue"),
        Script("""
        function autoRefresh() {
            setTimeout(function(){ window.location.reload(); }, 2000);
        }
        autoRefresh();
        """),
        Div(
            Safe("This is a live transcription and visual description of the Al Jazeera English channel. No data is stored. You can view the code at <a href='https://github.com/aastroza/tvtxt' target='_blank'>https://github.com/aastroza/tvtxt</a>"),
            _class="disclaimer"
        ),
        _class="script-page"
    )

serve()