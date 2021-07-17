import scrapy


class QuotesSpider(scrapy.Spider):
    name = "cve"

    def start_requests(self):
        urls = [
            'https://ubuntu.com/security/CVE-2016-1585',
            'https://ubuntu.com/security/CVE-2021-29973'
        ]
        ## We need to make sure that the URLs are unique because spiders on each
        ## URL work in parallel. To avoid conflicts in the future, we ensure that
        ## the list has unique URLs.
        urls = list(set(urls))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
 
    def process_row(self, row):
        """
        Processes an entire row of the status_table scraped from the CVE URL
        
        Parameters:
            row (list(str)): A list of strings
        
        Returns:
            release_status ((str, str)): Returns the release and the status associated to it
        
        Example:
            row:
                ['\n                        ', '\n                        ',
                '\n                      \n                        Upstream\n', 
                '\n                      Released\n                      ']
            
            release_status:
                ('Upstream', 'Released')

        """
        release = None
        status = ""
        for string in row:
            string = string.replace('\n',"") ## string_without_escape_seq
            processed_string = ' '.join(string.split()) ##string_without_extra_spaces
            if processed_string == "" or processed_string == None:
                continue
            if release is None:
                release = processed_string
            else:
                status = status + " " + processed_string
        status = status.lstrip().rstrip()
        release_status = (release, status)
        return release_status

    def parse(self, response):
        page_divs = response.css('div.wrapper').css('section.p-strip').css('div.row').css('div.col-9')
        status_table = page_divs[1].css('table.cve-table').css('tbody').css('tr')
        note_table = page_divs[2].css('table').css('tr')
        
        for row in status_table:
            processed_tuple = self.process_row(row.css('td::text').getall())
            if processed_tuple[0] is None:
                continue
            if processed_tuple[0].startswith('Upstream'):
                ## If the processed_tuple is Upstream, it refers to the first row of
                ## status_table. Its only the first row that has the package name and 
                ## hence we extract that using the underneath commands.
                package_list = row.css('td')[0].css('a::text').getall()
                print(('package_list',package_list))
            print(processed_tuple)
        
        ## For Note Section ##
        print(note_table[1].css('td').css('pre::text').get())