import train_data

if __name__ == "__main__":
    base_path = "generation/llama-3.2-11b-vision-instruct/imagenet-train"

    for name in ["new-prompts-1280k", "new-prompts-1280k-seed-110326"]:
        paths = [f"{base_path}/{name}-{i}" for i in range(16)]
        out_path = f"{base_path}/{name}"
        train_data.join_data(paths, out_path)
