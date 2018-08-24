#!/usr/bin/env python
from rftool.rf import get_args, main


if __name__ == '__main__':
    args = get_args()
    main(args.file, args.cv)
