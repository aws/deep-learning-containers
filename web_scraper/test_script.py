import scrapy
from scrapy.crawler import CrawlerRunner
## crochet plays with Twisted's Reactors to avoid ReactorNotRestartable error
from crochet import setup 
setup()

def run_spider(spiderClass, *args, **kwargs):
    crawler = CrawlerRunner()
    crawler.crawl(spiderClass, *args, **kwargs)

######################## How to run #####################
# from test_script import run_spider
# from web_scraper.spiders.cve_spiders import CveSpider
# run_spider(CveSpider, url_csv_string="https://ubuntu.com/security/CVE-2016-1585,https://ubuntu.com/security/CVE-2021-29973")
#########################################################
