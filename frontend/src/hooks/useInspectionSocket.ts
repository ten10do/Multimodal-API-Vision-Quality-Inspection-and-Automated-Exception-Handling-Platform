import { useCallback, useEffect, useRef, useState } from "react";
import type { WsConnectionState, WsEvent } from "../types";
import { parseWsEvent, pushDeduped } from "../utils/transforms";
import { DEFAULT_WS_URL, InspectionSocket } from "../ws/socket";

export const MAX_LIVE_EVENTS = 100; // bounded live list (4H)

export interface LiveInspectionState {
  events: WsEvent[];
  state: WsConnectionState;
  reconnect: () => void;
}

/** Live WS subscription with bounded list, dedup, reconnect + reconciliation.
 *  Events are either inspection.* or review.* notifications (5I); the DB is
 *  the source of truth and REST reconciliation runs after every (re)connect. */
export function useInspectionSocket(onReconcile: () => void): LiveInspectionState {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [state, setState] = useState<WsConnectionState>("connecting");
  const socketRef = useRef<InspectionSocket | null>(null);
  const onReconcileRef = useRef(onReconcile);
  onReconcileRef.current = onReconcile;

  useEffect(() => {
    const socket = new InspectionSocket(DEFAULT_WS_URL(), {
      onMessage: (raw: unknown) => {
        const ev = parseWsEvent(raw);
        if (ev) setEvents((prev) => pushDeduped(prev, ev, MAX_LIVE_EVENTS));
      },
      onStateChange: (s: WsConnectionState) => {
        setState(s);
        if (s === "connected" || s === "reconnected") {
          // reconciliation: REST is the source of truth (4F)
          void onReconcileRef.current();
        }
      },
    });
    socketRef.current = socket;
    socket.connect();
    return () => socket.disconnect();
  }, []);

  const reconnect = useCallback(() => {
    socketRef.current?.connect();
  }, []);

  return { events, state, reconnect };
}
