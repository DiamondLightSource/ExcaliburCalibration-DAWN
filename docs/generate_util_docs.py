import sys
import inspect

from excaliburcalibrationdawn import util


def main():

    functions = [name for name, _ in inspect.getmembers(util,
                                                        inspect.isfunction)]

    lines = ["Array Utilities\n",
             "===============\n",
             "\n",
             ".. module:: excaliburcalibrationdawn.util\n"
             "\n"]

    for function in functions:
        lines.append(".. autofunction:: {}\n".format(function))

    with open("arch/util.rst", "w") as docs_file:
        docs_file.writelines(lines)


if __name__ == "__main__":
    sys.exit(main())
