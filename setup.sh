# 建立一個資料夾來存放 NLTK 資料
mkdir -p /home/appuser/.nltk_data/

# 使用 Python 來執行下載指令，並將資料下載到我們指定的資料夾中
python -c "import nltk; nltk.download('punkt_tab', download_dir='/home/appuser/.nltk_data/'); nltk.download('punkt', download_dir='/home/appuser/.nltk_data/'); nltk.download('stopwords', download_dir='/home/appuser/.nltk_data/'); nltk.download('wordnet', download_dir='/home/appuser/.nltk_data/')"
