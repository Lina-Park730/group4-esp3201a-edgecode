"""Entry point for the sizes stage. Global options belong on run_pipeline.py."""
import sys
from edgecode.workflow import main
if __name__ == '__main__':
    sys.argv.insert(1, 'sizes')
    main()
