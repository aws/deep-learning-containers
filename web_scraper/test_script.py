import scrapy
from scrapy.crawler import CrawlerRunner
## crochet plays with Twisted's Reactors to avoid ReactorNotRestartable error
from crochet import setup 
setup()

def run_spider(spiderClass):
    crawler = CrawlerRunner()
    crawler.crawl(spiderClass)

######################## How to run #####################
# from test_script import run_spider
# from web_scraper.spiders.cve_spiders import CveSpider
# run_spider(CveSpider)
#########################################################
