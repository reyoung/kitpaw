import { stdout } from "node:process";
import { Input, Markdown, SelectList, SettingsList, Text, TUI } from "../../ref/pi-mono/packages/tui/src/index.ts";
import { HarnessTerminal } from "./terminal-harness.ts";

type ScenarioName =
  | "text-basic"
  | "input-focus"
  | "select-list-basic"
  | "overlay-centered"
  | "overlay-top-left"
  | "overlay-bottom-right"
  | "overlay-visible-rule"
  | "overlay-width-percent-min"
  | "overlay-short-content"
  | "overlay-style-reset"
  | "markdown-list"
  | "markdown-explicit-link"
  | "markdown-blockquote-wrap"
  | "markdown-table"
  | "markdown-code-fence"
  | "markdown-style-reset"
  | "settings-list-basic"
  | "settings-list-submenu-open"
  | "tui-middle-line-change"
  | "tui-clear-then-content"
  | "tui-style-reset";

class MutableComponent {
  lines: string[] = [];
  render(_width: number): string[] {
    return this.lines;
  }
  invalidate(): void {}
}

class StaticComponent {
  constructor(private readonly lines: string[]) {}
  render(_width: number): string[] {
    return this.lines;
  }
  invalidate(): void {}
}

const selectTheme = {
  selectedPrefix: (text: string): string => text,
  selectedText: (text: string): string => text,
  description: (text: string): string => text,
  scrollInfo: (text: string): string => text,
  noMatch: (text: string): string => text,
};

const markdownTheme = {
  heading: (text: string): string => text,
  link: (text: string): string => text,
  linkUrl: (text: string): string => text,
  code: (text: string): string => text,
  codeBlock: (text: string): string => text,
  codeBlockBorder: (text: string): string => text,
  quote: (text: string): string => text,
  quoteBorder: (text: string): string => text,
  hr: (text: string): string => text,
  listBullet: (text: string): string => text,
  bold: (text: string): string => text,
  italic: (text: string): string => text,
  strikethrough: (text: string): string => text,
  underline: (text: string): string => text,
};

const styledMarkdownTheme = {
  heading: (text: string): string => text,
  link: (text: string): string => text,
  linkUrl: (text: string): string => `\x1b[2m${text}\x1b[0m`,
  code: (text: string): string => `\x1b[33m${text}\x1b[0m`,
  codeBlock: (text: string): string => text,
  codeBlockBorder: (text: string): string => text,
  quote: (text: string): string => `\x1b[3m${text}\x1b[0m`,
  quoteBorder: (text: string): string => text,
  hr: (text: string): string => text,
  listBullet: (text: string): string => text,
  bold: (text: string): string => `\x1b[1m${text}\x1b[0m`,
  italic: (text: string): string => `\x1b[3m${text}\x1b[0m`,
  strikethrough: (text: string): string => text,
  underline: (text: string): string => `\x1b[4m${text}\x1b[0m`,
};

const settingsTheme = {
  label: (text: string, _selected: boolean): string => text,
  value: (text: string, _selected: boolean): string => text,
  description: (text: string): string => text,
  cursor: "→ ",
  hint: (text: string): string => text,
};

async function runScenario(name: ScenarioName): Promise<void> {
  const terminal = new HarnessTerminal(name === "select-list-basic" ? 80 : 20, 6);
  const tui = new TUI(terminal);

  if (name === "text-basic") {
    tui.addChild(new Text("hello"));
  } else if (name === "input-focus") {
    const input = new Input();
    input.focused = true;
    input.setValue("hello");
    tui.addChild(input);
  } else if (name === "select-list-basic") {
    const list = new SelectList(
      [
        { value: "short", label: "short", description: "short description" },
        {
          value: "very-long-command-name-that-needs-truncation",
          label: "very-long-command-name-that-needs-truncation",
          description: "long description",
        },
      ],
      5,
      selectTheme,
    );
    tui.addChild(list);
  } else if (name === "overlay-centered") {
    tui.addChild(new Text("Line 1\nLine 2\nLine 3"));
    tui.showOverlay(new Text("OVERLAY"), { anchor: "center", width: 10 });
  } else if (name === "overlay-top-left") {
    tui.addChild(new Text("base"));
    tui.showOverlay(new Text("TOP-LEFT"), { anchor: "top-left", width: 10 });
  } else if (name === "overlay-bottom-right") {
    tui.addChild(new Text("base"));
    tui.showOverlay(new Text("BTM-RIGHT"), { anchor: "bottom-right", width: 10 });
  } else if (name === "overlay-visible-rule") {
    tui.addChild(new Text("base"));
    tui.showOverlay(new Text("HIDDEN"), {
      anchor: "center",
      width: 10,
      visible: (columns: number, _rows: number): boolean => columns >= 30,
    });
  } else if (name === "overlay-width-percent-min") {
    tui.addChild(new Text("base"));
    tui.showOverlay(new Text("XXXXXXXXXXXXXXX"), {
      anchor: "center",
      width: "50%",
      minWidth: 12,
    });
  } else if (name === "overlay-short-content") {
    tui.addChild(new Text("Line 1\nLine 2\nLine 3"));
    tui.showOverlay(new Text("OVERLAY_TOP\nOVERLAY_MID\nOVERLAY_BOT"));
  } else if (name === "overlay-style-reset") {
    const component = new MutableComponent();
    component.lines = ["\x1b[3mXXXXXXXXXXXXXXXXXXXX\x1b[23m", "INPUT"];
    tui.addChild(component);
    tui.showOverlay(new Text("OVR"), { row: 0, col: 5, width: 3 });
  } else if (name === "markdown-list") {
    tui.addChild(
      new Markdown(
        "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
        0,
        0,
        markdownTheme,
      ),
    );
  } else if (name === "markdown-explicit-link") {
    tui.addChild(
      new Markdown(
        "[click here](https://example.com)",
        0,
        0,
        markdownTheme,
      ),
    );
  } else if (name === "markdown-blockquote-wrap") {
    tui.addChild(
      new Markdown(
        "> This is a very long blockquote line that should wrap to multiple lines when rendered",
        0,
        0,
        markdownTheme,
      ),
    );
  } else if (name === "markdown-table") {
    tui.addChild(
      new Markdown(
        "| Name | Value |\n| --- | --- |\n| Foo | Bar |\n| Baz | Qux |",
        0,
        0,
        markdownTheme,
      ),
    );
  } else if (name === "markdown-code-fence") {
    tui.addChild(
      new Markdown(
        "```py\nprint('hello')\nprint('world')\n```",
        0,
        0,
        markdownTheme,
      ),
    );
  } else if (name === "markdown-style-reset") {
    tui.addChild(
      new Markdown(
        "This is thinking with `inline code`",
        1,
        0,
        styledMarkdownTheme,
        {
          color: (text: string): string => `\x1b[90m${text}\x1b[0m`,
          italic: true,
        },
      ),
    );
    tui.addChild(new Text("INPUT"));
  } else if (name === "settings-list-basic") {
    tui.addChild(
      new SettingsList(
        [
          { id: "theme", label: "Theme", currentValue: "light", description: "UI theme", values: ["light", "dark"] },
          { id: "model", label: "Model", currentValue: "gpt-5" },
        ],
        5,
        settingsTheme,
        () => {},
        () => {},
      ),
    );
  } else if (name === "settings-list-submenu-open") {
    const settings = new SettingsList(
      [
        {
          id: "theme",
          label: "Theme",
          currentValue: "light",
          submenu: (currentValue: string, _done: (selectedValue?: string) => void) =>
            new StaticComponent([`submenu:${currentValue}`]),
        },
      ],
      5,
      settingsTheme,
      () => {},
      () => {},
    );
    settings.handleInput("\r");
    tui.addChild(settings);
  } else if (name === "tui-middle-line-change") {
    const component = new MutableComponent();
    component.lines = ["Header", "Working...", "Footer"];
    tui.addChild(component);
    tui.start();
    await terminal.flush();
    component.lines = ["Header", "Working /", "Footer"];
    tui.requestRender();
    await terminal.flush();
    stdout.write(JSON.stringify(terminal.snapshot()));
    tui.stop();
    return;
  } else if (name === "tui-clear-then-content") {
    const component = new MutableComponent();
    component.lines = ["Line 0", "Line 1", "Line 2"];
    tui.addChild(component);
    tui.start();
    await terminal.flush();
    component.lines = [];
    tui.requestRender();
    await terminal.flush();
    component.lines = ["New Line 0", "New Line 1"];
    tui.requestRender();
    await terminal.flush();
    stdout.write(JSON.stringify(terminal.snapshot()));
    tui.stop();
    return;
  } else if (name === "tui-style-reset") {
    const component = new MutableComponent();
    component.lines = ["\x1b[3mItalic", "Plain"];
    tui.addChild(component);
    tui.start();
    await terminal.flush();
    stdout.write(JSON.stringify(terminal.snapshot()));
    tui.stop();
    return;
  }

  tui.start();
  await terminal.flush();
  stdout.write(JSON.stringify(terminal.snapshot()));
  tui.stop();
}

async function main(): Promise<void> {
  const scenario = (process.argv[2] ?? "text-basic") as ScenarioName;
  await runScenario(scenario);
}

void main();
