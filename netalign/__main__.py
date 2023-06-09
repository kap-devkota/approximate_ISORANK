"""
Network Alignment Tools; Includes Code for running DUOMUNDO and Approximate ISORANK
"""
import argparse
import os
import sys
from typing import Union

from .duomundo.duomundo_main import DuoMundoArgs
from .approx_isorank.aisorank import ApproxIsorankArgs

NetPackArguments = Union[
        DuoMundoArgs,
        ApproxIsorankArgs
    ]


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CitationAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    from . import __version__

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v", "--version", action="version", version="NetAlign " + __version__
    )
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit",
    )

    subparsers = parser.add_subparsers(title="NetAlign Commands", dest="cmd")
    subparsers.required = True

    from .duomundo import duomundo_main
    from .approx_isorank import aisorank
    

    modules = {
        "duomundo": duomundo_main,
        "isorank": aisorank,
    }
    
    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args: NetPackArguments = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()