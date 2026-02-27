from SoccerNet.Downloader import SoccerNetDownloader

# This will create a SoccerNet/ folder in your current directory
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./SoccerNet")

# Download the jersey-2023 dataset (train + test)
mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train", "test"])