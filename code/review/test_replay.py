"""Synthetic construction tests; no benchmark text or model calls."""
from p0_common import replay_after_target,final_only_history,user_sequence_hash
messages=[{'role':'system','content':'Synthetic test.'}]
for i in range(3):messages += [{'role':'user','content':f'user {i}'},{'role':'assistant','content':f'answer number {i}'}]
def call(h,m):return dict(message='fresh',returned_model=m,response_id='synthetic')
for target in [0,1]:
    for replace in [False,True]:
        h,g=replay_after_target(messages,'synthetic',target,replace,call)
        assert len(g)==2-target and user_sequence_hash(h)==user_sequence_hash(messages)
for condition in ['null_replay','short_neutralization','length_matched_neutralization']:
    h,g,_=final_only_history(messages,'synthetic',condition,call)
    assert len(g)==1 and user_sequence_hash(h)==user_sequence_hash(messages)
print('REPLAY_CONSTRUCTION_TESTS: PASS (7 synthetic cases; zero API calls)')
import tempfile,json,subprocess,sys
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory)
    manifest=folder/'manifest.jsonl'
    manifest.write_text(json.dumps(dict(pair_id='synthetic-pair',trajectory_id='synthetic-trajectory',task='actions',model='synthetic',source_instance_id='synthetic',target_turn=0,original_messages=messages,user_sequence_hash=user_sequence_hash(messages)))+'\n',encoding='utf8')
    for mode in ['p0a','p0b']:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_null_replay.py')),'--mode',mode,'--manifest',str(manifest),'--output-dir',str(folder/'outputs')],check=True)

subprocess.run([sys.executable, str(Path(__file__).with_name('test_code_capture.py'))], check=True)
