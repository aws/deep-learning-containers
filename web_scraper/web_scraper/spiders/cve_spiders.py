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

    def process_row(self, row):
        release = None
        status = ""
        for string in row:
            truncated_string = string.lstrip('\n ').rstrip('\n ')
            if truncated_string == "" or truncated_string == None:
                continue
            if release is None:
                release = truncated_string
            else:
                status = status + " " + truncated_string
        return (release, status)

    def parse(self, response):
        page_divs = response.css('div.wrapper').css('section.p-strip').css('div.row').css('div.col-9')
        status_table = page_divs[1].css('table.cve-table').css('tbody').css('tr')
        note_table = page_divs[2].css('table').css('tr')
        
        ## For any rows ##
        for row in status_table:
            processed_tuple = self.process_row(row.css('td::text').getall())
            if processed_tuple[0] is None:
                continue
            if processed_tuple[0] == 'Upstream':
                package_list = row.css('td')[0].css('a::text').getall()
                print(('package_list',package_list))
            print(self.process_row(row.css('td::text').getall()))
        
        ## For Note Section ##
        print(note_table[1].css('td').css('pre::text').get())