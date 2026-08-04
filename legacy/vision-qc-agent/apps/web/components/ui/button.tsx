import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#233d2f]",
  {
    variants: {
      variant: {
        default: "bg-[#233d2f] text-white hover:bg-[#182a21]",
        accent: "bg-[#d8ff65] text-[#18201d] hover:bg-[#c9f04f]",
        outline: "border bg-white text-[#233d2f] hover:bg-[#f1f5f3]",
        danger: "bg-[#d9433f] text-white hover:bg-[#bd3431]",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Button({
  className,
  variant,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants>) {
  return (
    <button className={cn(buttonVariants({ variant }), className)} {...props} />
  );
}
