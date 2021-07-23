import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
## crochet plays with Twisted's Reactors to avoid ReactorNotRestartable error
from crochet import setup 
setup()

def get_crawler(storage_file_path=None):
    if storage_file_path is not None:
        s = get_project_settings()
        s['FEED_FORMAT'] = 'json'
        s['FEED_URI'] = storage_file_path
        s['FEED_EXPORT_INDENT'] = 4
        return CrawlerRunner(s)

    return CrawlerRunner()

def run_spider(spiderClass, storage_file_path=None, *args, **kwargs):
    crawler = get_crawler(storage_file_path)
    crawler.crawl(spiderClass, *args, **kwargs)

######################## How to run #####################
# from test_script import run_spider
# from web_scraper.spiders.cve_spiders import CveSpider
# run_spider(CveSpider,storage_file_path='cve_data.json', url_csv_string="https://ubuntu.com/security/CVE-2016-1585,https://ubuntu.com/security/CVE-2021-29973")
#########################################################