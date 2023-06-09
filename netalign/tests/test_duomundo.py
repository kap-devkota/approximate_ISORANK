import os
import shutil
import subprocess as sp


class TestDuomundo:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()
        os.makedirs("./tmp-duomundo-dir/", exist_ok = True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp-duomundo-dir/")

    def _run_command(self, cmd):
        proc = sp.Popen(cmd.split())
        proc.wait()
        assert not proc.returncode

    def test_duomundo(self):
        cmd = "netalign duomundo --ppiA ./netalign/tests/fly.s.tsv --ppiB ./netalign/tests/rat.s.tsv --nameA fly --nameB rat " \
              "--thres_dsd_dist 10 --dsd_A_dist ./tmp-duomundo-dir/fly-dsd-dist.npy --dsd_B_dist ./tmp-duomundo-dir/rat-dsd-dist.npy " \
              "--json_A ./tmp-duomundo-dir/fly.json --json_B ./tmp-duomundo-dir/rat.json --svd_AU ./tmp-duomundo-dir/fly-left-svd.npy " \
              "--svd_BU ./tmp-duomundo-dir/rat-left-svd.npy --svd_AV ./tmp-duomundo-dir/fly-right-svd.npy --svd_BV ./tmp-duomundo-dir/rat-right-svd.npy " \
              "--svd_r 100 --landmarks_a_b ./netalign/tests/fly-rat.tsv --compute_isorank " \
              "--model ./tmp-duomundo-dir/fly-rat.model --svd_dist_a_b ./tmp-duomundo-dir/fly-rat-svd-dist.npy --compute_go_eval --kA 10 --kB 10 " \
              "--metrics top-1-acc --output_file ./tmp-duomundo-dir/test-fly-rat.tsv --go_A ./data/go/fly.output.mapping.gaf " \
              "--go_B ./data/go/rat.output.mapping.gaf"
        self._run_command(cmd)