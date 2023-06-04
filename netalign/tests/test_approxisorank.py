import os
import shutil
import subprocess as sp


class TestAIsorank:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()
        os.makedirs("./tmp-aisorank-dir/", exist_ok = True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp-aisorank-dir/")

    def _run_command(self, cmd):
        proc = sp.Popen(cmd.split())
        proc.wait()
        assert not proc.returncode

    def test_aisorank_r0(self):
        cmd = "netalign isorank --net1 ./netalign/tests/fly.s.tsv --net2 ./netalign/tests/rat.s.tsv --rblast ./netalign/tests/fly-rat.tsv --alpha 0.6 --niter 0 --npairs 50 --output ./tmp-aisorank-dir/human-mouse-0-50-0.6.tsv"
        self._run_command(cmd)