export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return <div className="state-block loading">{label}</div>;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-block error" role="alert">
      <div className="state-title">数据获取失败</div>
      <div className="state-message">{message}</div>
      {onRetry ? (
        <button className="btn" onClick={onRetry}>
          重试
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({ message = "暂无数据" }: { message?: string }) {
  return <div className="state-block empty">{message}</div>;
}
