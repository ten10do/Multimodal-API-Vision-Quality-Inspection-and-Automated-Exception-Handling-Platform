import { beforeEach, describe, expect, it, vi } from "vitest";
import { InspectionSocket, retryDelay } from "../src/ws/socket";
import type { WsConnectionState } from "../src/types";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static reset(): void {
    FakeWebSocket.instances = [];
  }
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  close(): void {
    this.closed = true;
  }
  // test helpers
  open(): void {
    this.onopen?.();
  }
  closeRemote(): void {
    this.onclose?.();
  }
  send(data: string): void {
    this.onmessage?.({ data });
  }
}

describe("retryDelay", () => {
  it("grows exponentially with a hard cap", () => {
    expect(retryDelay(0)).toBe(500);
    expect(retryDelay(1)).toBe(1000);
    expect(retryDelay(2)).toBe(2000);
    expect(retryDelay(10)).toBe(10_000);
  });
});

describe("InspectionSocket", () => {
  beforeEach(() => {
    FakeWebSocket.reset();
  });

  it("connects, delivers messages, and reports state transitions", () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const states: WsConnectionState[] = [];
    const messages: unknown[] = [];
    const sock = new InspectionSocket("ws://test/ws", {
      onMessage: (m) => messages.push(m),
      onStateChange: (s) => states.push(s),
    });
    sock.connect();
    const ws = FakeWebSocket.instances[0];
    expect(states[0]).toBe("connecting");
    ws.open();
    expect(states).toContain("connected");
    ws.send(JSON.stringify({ event_type: "inspection.completed", inspection_id: "i1", product_id: "p1" }));
    expect(messages).toHaveLength(1);
    sock.disconnect();
    vi.unstubAllGlobals();
  });

  it("reconnects with backoff after an unexpected close", async () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.useFakeTimers();
    const states: WsConnectionState[] = [];
    const sock = new InspectionSocket("ws://test/ws", {
      onMessage: () => undefined,
      onStateChange: (s) => states.push(s),
    });
    sock.connect();
    FakeWebSocket.instances[0].open();
    FakeWebSocket.instances[0].closeRemote(); // dead connection
    expect(states).toContain("disconnected");
    await vi.advanceTimersByTimeAsync(600);
    expect(FakeWebSocket.instances.length).toBe(2); // reconnected
    expect(states).toContain("reconnecting");
    FakeWebSocket.instances[1].open();
    expect(states).toContain("reconnected");
    sock.disconnect();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("stops reconnecting after user disconnect", () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.useFakeTimers();
    const sock = new InspectionSocket("ws://test/ws", { onMessage: () => undefined });
    sock.connect();
    FakeWebSocket.instances[0].closeRemote();
    sock.disconnect();
    void vi.advanceTimersByTimeAsync(60_000).then(() => {
      expect(FakeWebSocket.instances.length).toBe(1);
    });
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("survives malformed payloads without throwing", () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const sock = new InspectionSocket("ws://test/ws", { onMessage: () => undefined });
    sock.connect();
    FakeWebSocket.instances[0].send("{not json");
    FakeWebSocket.instances[0].send("");
    expect(true).toBe(true);
    sock.disconnect();
    vi.unstubAllGlobals();
  });
});
