from ingest_data import download_data


def main():
    print("Starting pipeline...")
    download_data()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()