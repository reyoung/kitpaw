import { stdin, stdout } from "node:process";
import { HarnessTerminal } from "./terminal-harness.ts";

type Request = {
  writes: string[];
  columns?: number;
  rows?: number;
};

const chunks: Buffer[] = [];
stdin.on("data", (chunk) => chunks.push(Buffer.from(chunk)));

stdin.on("end", async () => {
  const req = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Request;
  const terminal = new HarnessTerminal(req.columns ?? 80, req.rows ?? 24);
  for (const write of req.writes) {
    terminal.write(write);
  }
  await terminal.flush();
  stdout.write(JSON.stringify(terminal.snapshot()));
});
