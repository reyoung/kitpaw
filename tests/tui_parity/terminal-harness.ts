import { Terminal as XtermTerminal } from "../../ref/pi-mono/node_modules/@xterm/headless/lib-headless/xterm-headless.js";

export type TerminalSnapshot = {
  viewport: string[];
  cursor: { x: number; y: number };
  scrollback: string[];
  italic: number[][];
};

export class HarnessTerminal {
  private readonly xterm: XtermTerminal;
  private readonly cols: number;
  private readonly rows: number;

  constructor(columns = 80, rows = 24) {
    this.cols = columns;
    this.rows = rows;
    this.xterm = new XtermTerminal({
      cols: columns,
      rows,
      disableStdin: true,
      allowProposedApi: true,
    });
  }

  start(onInput: (data: string) => void, onResize: () => void): void {
    void onInput;
    void onResize;
  }

  stop(): void {}

  async drainInput(): Promise<void> {}

  write(data: string): void {
    this.xterm.write(data);
  }

  get columns(): number {
    return this.cols;
  }

  get rows(): number {
    return this.rows;
  }

  get kittyProtocolActive(): boolean {
    return true;
  }

  moveBy(lines: number): void {
    if (lines > 0) {
      this.write(`\x1b[${lines}B`);
    } else if (lines < 0) {
      this.write(`\x1b[${-lines}A`);
    }
  }

  hideCursor(): void {
    this.write("\x1b[?25l");
  }

  showCursor(): void {
    this.write("\x1b[?25h");
  }

  clearLine(): void {
    this.write("\x1b[K");
  }

  clearFromCursor(): void {
    this.write("\x1b[J");
  }

  clearScreen(): void {
    this.write("\x1b[2J\x1b[H");
  }

  setTitle(title: string): void {
    this.write(`\x1b]0;${title}\x07`);
  }

  async flush(): Promise<void> {
    await new Promise<void>((resolve) => {
      this.xterm.write("", () => resolve());
    });
  }

  snapshot(): TerminalSnapshot {
    const viewport: string[] = [];
    const italic: number[][] = [];
    const buffer = this.xterm.buffer.active;
    for (let i = 0; i < this.rows; i++) {
      const line = buffer.getLine(buffer.viewportY + i);
      viewport.push(line ? line.translateToString(true) : "");
      const rowStyles: number[] = [];
      for (let col = 0; col < this.cols; col++) {
        const cell = line?.getCell(col);
        rowStyles.push(cell ? cell.isItalic() : 0);
      }
      italic.push(rowStyles);
    }
    const scrollback: string[] = [];
    for (let i = 0; i < buffer.length; i++) {
      const line = buffer.getLine(i);
      scrollback.push(line ? line.translateToString(true) : "");
    }
    return {
      viewport,
      cursor: { x: buffer.cursorX, y: buffer.cursorY },
      scrollback,
      italic,
    };
  }
}
