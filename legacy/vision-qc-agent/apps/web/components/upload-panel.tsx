"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ImageUp, LoaderCircle } from "lucide-react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { createInspection } from "@/lib/api";

const schema = z.object({
  product_code: z.string().trim().min(1, "请输入产品编码").max(100),
  batch_code: z.string().trim().min(1, "请输入批次").max(100),
  image: z
    .custom<FileList>(
      (value) => typeof FileList !== "undefined" && value instanceof FileList,
      "请选择一张图片",
    )
    .refine((files) => files.length === 1, "请选择一张图片")
    .refine(
      (files) =>
        files.length === 0 ||
        ["image/jpeg", "image/png", "image/webp"].includes(files[0].type),
      "仅支持 JPG、PNG、WEBP",
    )
    .refine(
      (files) => files.length === 0 || files[0].size <= 10 * 1024 * 1024,
      "图片不得超过 10MB",
    ),
});

type UploadValues = z.infer<typeof schema>;

export function UploadPanel() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
  } = useForm<UploadValues>({ resolver: zodResolver(schema) });
  const selectedFiles = useWatch({ control, name: "image" });
  const selectedFile = selectedFiles?.[0];
  const mutation = useMutation({
    mutationFn: createInspection,
    onSuccess: async (inspection) => {
      await queryClient.invalidateQueries({ queryKey: ["inspections"] });
      await queryClient.invalidateQueries({ queryKey: ["stats"] });
      router.push(`/inspections/${inspection.id}`);
    },
  });

  const submit = (values: UploadValues) => {
    const form = new FormData();
    form.set("product_code", values.product_code);
    form.set("batch_code", values.batch_code);
    form.set("image", values.image[0]);
    mutation.mutate(form);
  };

  return (
    <form
      onSubmit={handleSubmit(submit)}
      className="grid gap-4 md:grid-cols-[1fr_1fr_1.3fr_auto]"
    >
      <label className="grid gap-1.5 text-xs font-semibold text-slate-600">
        产品编码
        <input
          {...register("product_code")}
          placeholder="例如：AX-240"
          className="h-11 rounded-lg border bg-white px-3 text-sm font-normal outline-none focus:border-[#233d2f]"
        />
        {errors.product_code && (
          <span className="text-red-600">{errors.product_code.message}</span>
        )}
      </label>
      <label className="grid gap-1.5 text-xs font-semibold text-slate-600">
        生产批次
        <input
          {...register("batch_code")}
          placeholder="例如：B20260730"
          className="h-11 rounded-lg border bg-white px-3 text-sm font-normal outline-none focus:border-[#233d2f]"
        />
        {errors.batch_code && (
          <span className="text-red-600">{errors.batch_code.message}</span>
        )}
      </label>
      <label className="grid gap-1.5 text-xs font-semibold text-slate-600">
        质检图片
        <span className="flex h-11 cursor-pointer items-center gap-2 rounded-lg border bg-white px-3 text-sm font-normal text-slate-500 hover:bg-slate-50">
          <ImageUp size={16} />
          选择 JPG / PNG / WEBP
          <input
            {...register("image")}
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            className="sr-only"
          />
        </span>
        {errors.image && (
          <span className="text-red-600">{errors.image.message as string}</span>
        )}
        {selectedFile && (
          <ImagePreview
            key={`${selectedFile.name}-${selectedFile.size}-${selectedFile.lastModified}`}
            file={selectedFile}
          />
        )}
      </label>
      <Button
        variant="accent"
        className="h-11 self-end whitespace-nowrap"
        disabled={mutation.isPending}
      >
        {mutation.isPending ? (
          <LoaderCircle className="animate-spin" size={16} />
        ) : (
          <ImageUp size={16} />
        )}
        开始质检
      </Button>
      {mutation.error && (
        <p className="text-sm text-red-600 md:col-span-4">
          {mutation.error.message}
        </p>
      )}
    </form>
  );
}

function ImagePreview({ file }: { file: File }) {
  const [previewUrl] = useState(() => URL.createObjectURL(file));
  useEffect(() => () => URL.revokeObjectURL(previewUrl), [previewUrl]);
  return (
    <span className="mt-1 flex items-center gap-2 rounded-lg border bg-slate-50 p-2">
      <Image
        src={previewUrl}
        alt="待检测图片预览"
        width={48}
        height={48}
        unoptimized
        className="size-12 rounded-md object-cover"
      />
      <span className="truncate text-xs font-normal text-slate-600">
        {file.name}
      </span>
    </span>
  );
}
