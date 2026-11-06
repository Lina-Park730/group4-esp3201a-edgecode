"""Entry point for the evaluate stage. Global options belong on run_pipeline.py."""
import sys
from edgecode.workflow import main
if __name__ == '__main__':
    sys.argv.insert(1, 'evaluate')
    main()
