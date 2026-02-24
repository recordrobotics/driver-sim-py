from __future__ import annotations

import moderngl_window as mglw

from renderer import RealtimeRenderer


def main() -> None:
    mglw.run_window_config(RealtimeRenderer)


if __name__ == "__main__":
    main()
