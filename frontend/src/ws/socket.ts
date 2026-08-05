// WebSocket client with bounded-exponential reconnect (4F).
// Connection state is exposed so the UI can show it; after a reconnect the
// caller runs a REST reconciliation because PostgreSQL is the source of truth
// and the WS channel may have missed events while disconnected.

import type { WsConnectionState } from "../types";

export type { WsConnectionState } from "../types";

export interface SocketHandlers {
  onMessage: (raw: unknown) => void;
  onStateChange?: (state: WsConnectionState) => void;
}

export interface SocketController {
  connect: () => void;
  disconnect: () => void;
}

export const DEFAULT_WS_URL = (): string => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/api/v1/ws/inspections`;
};

const MAX_RETRY_DELAY_MS = 10_000;
const BASE_RETRY_DELAY_MS = 500;

export function retryDelay(attempt: number): number {
  return Math.min(BASE_RETRY_DELAY_MS * 2 ** Math.min(attempt, 6), MAX_RETRY_DELAY_MS);
}

export class InspectionSocket {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private closedByUser = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly url: string,
    private readonly handlers: SocketHandlers,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  disconnect(): void {
    this.closedByUser = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
    this.handlers.onStateChange?.("disconnected");
  }

  private open(): void {
    if (this.closedByUser) return;
    this.handlers.onStateChange?.(this.attempt === 0 ? "connecting" : "reconnecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      const wasReconnect = this.attempt > 0;
      this.attempt = 0;
      this.handlers.onStateChange?.(wasReconnect ? "reconnected" : "connected");
    };

    ws.onmessage = (ev) => {
      try {
        this.handlers.onMessage(JSON.parse(String(ev.data)));
      } catch {
        // malformed payload: ignore, never crash the page
      }
    };

    ws.onclose = () => {
      if (this.closedByUser) return;
      this.handlers.onStateChange?.("disconnected");
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      // close follows; nothing else to do here
    };
  }

  private scheduleReconnect(): void {
    if (this.closedByUser || this.timer) return;
    const delay = retryDelay(this.attempt);
    this.attempt += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.open();
    }, delay);
  }
}

/** React hook: connects once, surfaces state + latest events, and runs a
 * reconciliation callback after every (re)connect. */
export function createSocketController(
  url: string,
  handlers: SocketHandlers,
): SocketController {
  return new InspectionSocket(url, handlers);
}
