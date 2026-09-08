"""Run Bijendra's task from VS Code or a terminal."""
import sys

if __name__ == '__main__':
    try:
        from src.task2 import run
        run()
    except ModuleNotFoundError as exc:
        print(f'Missing Python package: {exc.name}. Run: python -m pip install -r requirements.txt',file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError) as exc:
        print(f'Task 2 stopped: {exc}\nPreviously saved outputs, if any, were not validated by this attempt.',file=sys.stderr)
        sys.exit(1)
