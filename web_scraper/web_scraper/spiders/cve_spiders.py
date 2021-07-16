import scrapy


class QuotesSpider(scrapy.Spider):
    name = "cve"

    def start_requests(self):
        urls = [
            'https://ubuntu.com/security/CVE-2016-1585',
            'https://ubuntu.com/security/CVE-2021-29973'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page_divs = response.css('div.wrapper').css('section.p-strip').css('div.row').css('div.col-9')
        status_table = page_divs[1].css('table.cve-table').css('tbody').css('tr')
        note_table = page_divs[2].css('table').css('tr')
        
        ## For any rows ##
        for row in status_table:
            print(row.css('td::text').getall())
        print(note_table[1].css('td').css('pre::text').get())