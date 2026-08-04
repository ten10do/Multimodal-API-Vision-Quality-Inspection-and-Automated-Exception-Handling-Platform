import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UploadPanel } from "@/components/upload-panel";

const mocks = vi.hoisted(() => ({
  createInspection: vi.fn(),
  push: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  createInspection: mocks.createInspection,
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mocks.push }),
}));

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <UploadPanel />
    </QueryClientProvider>,
  );
}

describe("upload panel", () => {
  beforeEach(() => {
    mocks.createInspection.mockReset();
    mocks.push.mockReset();
  });

  it("validates required product, batch and image fields", async () => {
    const user = userEvent.setup();
    renderPanel();
    await user.click(screen.getByRole("button", { name: "开始质检" }));
    expect(await screen.findByText("请输入产品编码")).toBeInTheDocument();
    expect(screen.getByText("请输入批次")).toBeInTheDocument();
    expect(screen.getByText("请选择一张图片")).toBeInTheDocument();
    expect(mocks.createInspection).not.toHaveBeenCalled();
  });

  it("rejects unsupported image types in the browser", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderPanel();
    const file = new File(["gif"], "part.gif", { type: "image/gif" });
    await user.upload(screen.getByLabelText(/质检图片/), file);
    await user.type(screen.getByLabelText("产品编码"), "AX-240");
    await user.type(screen.getByLabelText("生产批次"), "B-01");
    await user.click(screen.getByRole("button", { name: "开始质检" }));
    expect(
      await screen.findByText("仅支持 JPG、PNG、WEBP"),
    ).toBeInTheDocument();
  });

  it("shows a local image preview before submission", async () => {
    const user = userEvent.setup();
    renderPanel();
    const file = new File(["png"], "part.png", { type: "image/png" });
    await user.upload(screen.getByLabelText(/质检图片/), file);
    expect(
      await screen.findByRole("img", { name: "待检测图片预览" }),
    ).toHaveAttribute("src", "blob:test-preview");
    expect(screen.getByText("part.png")).toBeInTheDocument();
  });

  it("surfaces Provider errors returned by the API", async () => {
    const user = userEvent.setup();
    mocks.createInspection.mockRejectedValueOnce(
      new Error("百炼 Provider 不可用"),
    );
    renderPanel();
    await user.type(screen.getByLabelText("产品编码"), "AX-240");
    await user.type(screen.getByLabelText("生产批次"), "B-01");
    await user.upload(
      screen.getByLabelText(/质检图片/),
      new File(["png"], "part.png", { type: "image/png" }),
    );
    await user.click(screen.getByRole("button", { name: "开始质检" }));
    expect(await screen.findByText("百炼 Provider 不可用")).toBeInTheDocument();
  });
});
