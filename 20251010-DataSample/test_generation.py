import train_data

if __name__ == "__main__":
    config = train_data.GenerationConfig.default()

    config.data_range = (0, 2048)
    config.n_generated_tokens = 512

    train_data.generate_data(config)
