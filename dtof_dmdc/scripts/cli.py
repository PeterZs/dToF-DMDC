import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import click


@click.group(help='dToF-DMDC command line interface.')
def cli():
    pass

def main():
    from dtof_dmdc.scripts import infer, infer_baseline, eval_baseline
    cli.add_command(infer.main, name='infer')
    cli.add_command(infer_baseline.main, name='infer_baseline')
    cli.add_command(eval_baseline.main, name='eval_baseline')
    cli()


if __name__ == '__main__':
    main()