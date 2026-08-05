/// <reference types="vite/client" />

// Minimal node-like env access for vite.config.ts without @types/node.
declare const process: {
  env: { [key: string]: string | undefined };
};
