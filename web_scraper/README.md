# Webscraper

This module helps us with the process of scraping the web. The module has been built using [scrapy](https://docs.scrapy.org/en/latest/intro/overview.html#scrapy-at-a-glance), an application framework used for crawling web sites and extracting structured data.

As of now, the WebScraper has just 1 spider that is used to crawl the CVE URLs. 

## Setup:
Installing [scrapy](https://docs.scrapy.org/en/latest/intro/install.html).

## Running the Code:
To run the code, the following steps can be followed:

**Note, all the commands given below should be run from this directory i.e. `~/deep-learning-containers/web_scraper itself.**

1. Set the following environment variable:
`export SCRAPE_URL_LIST="https://ubuntu.com/security/CVE-2016-1585 https://ubuntu.com/security/CVE-2021-29973"`

The `SCRAPE_URL_LIST` environment variable consists of the list of space seperated CVE URLs to scrape. 

2. In the deep-learning-containers root folder, remove the file scraped_data using the following command `rm -r ../scraped_data.json`.

3. Start the crawl by using the following command `scrapy crawl cve -o ../scraped_data.json`

4. A JSON file name `scraped_data.json` will be created in the root folder of the deep-learning-container repository.

