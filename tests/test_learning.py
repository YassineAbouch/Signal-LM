import io
import shutil
import threading
import wave
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch

from signal_lm import learning, model_test
from signal_lm.data import load_signal
from signal_lm.data.schema import read_manifest
from signal_lm.signal import Signal
from signal_lm.web import Handler


def tone():
    values=(np.sin(2*np.pi*200*np.arange(8000)/8000)*8192).astype('<i2')
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(8000);wav.writeframes(values.tobytes())
    return stream.getvalue(),values.astype(np.float32)/32768


def save(root, **kwargs):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/ffprobe required for audio annotation integration')
    raw,_=tone()
    options=dict(zone='atelier',window_start=0,selection_start=.1,selection_end=.6,
                 label='impulsion',meaning='Observation utilisateur',root=root)
    options.update(kwargs)
    return learning.save_annotation(raw,'essai.wav',**options)


def test_annotation_preserves_samples_and_is_idempotent(tmp_path):
    result=save(tmp_path)
    assert result['created'] and not result['training_started']
    records=read_manifest(tmp_path/'manifest.jsonl')
    assert len(records)==1
    record=records[0]
    assert record.split=='unassigned' and not record.is_synthetic
    assert record.labels==['impulsion']
    assert record.provenance['review_status']=='unverified_user_annotation'
    assert record.provenance['acquisition_time_utc'] is None
    signal=load_signal(tmp_path,record)
    assert signal.sample_rate_hz==8000 and signal.units==('unspecified',)
    _,original=tone()
    np.testing.assert_array_equal(signal.samples[:,0],original[800:4800])
    assert np.sqrt(np.mean(signal.samples**2))==pytest.approx(.25/np.sqrt(2),abs=1e-4)
    assert save(tmp_path)['created'] is False
    assert len(read_manifest(tmp_path/'manifest.jsonl'))==1
    save(tmp_path,selection_start=.2,label='autre')
    records=read_manifest(tmp_path/'manifest.jsonl')
    assert len(records)==2 and records[0].group_id==records[1].group_id
    assert learning.stats(tmp_path)['unassigned']==2


def test_machine_group_cannot_change_for_same_acquisition(tmp_path):
    save(tmp_path,machine_id='fan01')
    with pytest.raises(ValueError,match='autre groupe'):
        save(tmp_path,machine_id='fan02')
    assert len(read_manifest(tmp_path/'manifest.jsonl'))==1


@pytest.mark.parametrize('changes',[{'label':''},{'meaning':' '},{'selection_start':-.1},
    {'selection_end':1.1},{'selection_end':.1},{'selection_end':.1001},
    {'selection_start':float('nan')}])
def test_invalid_annotation_never_creates_manifest(tmp_path,changes):
    with pytest.raises(ValueError):save(tmp_path,**changes)
    assert not (tmp_path/'manifest.jsonl').exists()


def test_inference_normalizes_copy_and_does_not_change_weights(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.gain=torch.nn.Parameter(torch.tensor(2.))
        def forward(self,x):return {'embedding':x.mean(dim=1)*self.gain}
    model=Model()
    payload={'input_contract':{'sample_rate_hz':256,'n_samples':64,'channel_names':['a','b'],'units':['g','g']},
             'normalization':{'mean':[1.,2.],'scale':[2.,4.]},'center':torch.tensor([0.,0.]),'threshold':7.}
    monkeypatch.setattr(model_test,'loaded_model',lambda:(model,payload))
    signal=Signal(np.tile([3.,6.],(64,1)),256,('a','b'),('g','g'))
    before=signal.samples.copy();weight=model.gain.detach().clone()
    result=model_test.predict(signal)
    assert result['score']==pytest.approx(8.) and result['above_threshold']
    assert result['confidence'] is None and not result['weights_updated']
    np.testing.assert_array_equal(signal.samples,before)
    assert torch.equal(weight,model.gain)
    for bad in [Signal(np.ones((63,2)),256),Signal(np.ones((64,1)),256),Signal(np.ones((64,2)),512)]:
        with pytest.raises(ValueError,match='incompatible'):model_test.predict(bad)
    with pytest.raises(ValueError,match='incompatible'):
        model_test.predict(Signal(np.ones((64,2)),256,('a','b'),('Pa','Pa')))


def test_learning_http_explicit_save_and_same_origin(tmp_path,monkeypatch):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/ffprobe required for audio annotation integration')
    monkeypatch.setattr(learning,'LEARNING_ROOT',tmp_path)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=HTTPConnection('127.0.0.1',server.server_port,timeout=20)
    try:
        for path in ['/learn','/learn.js','/test-model','/test-model.js','/api/learning/stats']:
            client.request('GET',path);response=client.getresponse();assert response.status==200;response.read()
        assert not (tmp_path/'manifest.jsonl').exists()
        raw,_=tone()
        client.request('POST','/api/learning/save',raw,{'Origin':'https://example.com'})
        response=client.getresponse();assert response.status==403;response.read()
        client.close();client=HTTPConnection('127.0.0.1',server.server_port,timeout=20)
        client.request('POST','/api/learning/save',raw,{'X-Audio-Name':'tone.wav','X-Learning-Label':'tone','X-Learning-Meaning':'test observation','X-Selection-End':'.5'})
        response=client.getresponse();body=response.read();assert response.status==200,body
        assert len(read_manifest(tmp_path/'manifest.jsonl'))==1
    finally:client.close();server.shutdown();server.server_close();thread.join()
