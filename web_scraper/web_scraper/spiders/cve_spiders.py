import scrapy


class QuotesSpider(scrapy.Spider):
    name = "cve" ## Spider name should be unique

    def start_requests(self):
        urls = [
            'https://ubuntu.com/security/CVE-2016-1585',
            'https://ubuntu.com/security/CVE-2021-29973'
        ]
        ## Parse function stores the data related to each CVE in a JSON format.
        ## For the data to be stored properly, make sure that the URLs are unique. 
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
        page_divisions = response.css('div.wrapper').css('section.p-strip').css('div.row').css('div.col-9')
        status_table = page_divisions[1].css('table.cve-table').css('tbody').css('tr')
        note_table = page_divisions[2].css('table').css('tr')
        
        processed_data = {
            'URL':response.url,
            'status_tables':[],
            'notes': []
        }

        table_data = None
        for row in status_table:
            ## The td tag in each row has the important text stored in it.
            ## Sometimes, td has some text in small fonts. That text is simply appended at 
            ## the end to fetch and preserve maximum amount of data.
            processed_tuple = self.process_row(row.css('td::text').getall() + row.css('td').css('small::text').getall())
            if processed_tuple[0] is None:
                continue
            if processed_tuple[0].startswith('Upstream'):
                ## If the processed_tuple is Upstream, it refers to the first row of
                ## status_table. Its only the first row that has the package name and 
                ## hence we extract that using the underneath commands.
                if table_data is not None:
                    processed_data['status_tables'].append(table_data)
                table_data = {}
                package_list = row.css('td')[0].css('a::text').getall()
                table_data['packages'] = package_list
            if 'release_states' not in table_data:
                table_data['release_states'] = []
            table_data['release_states'].append(list(processed_tuple))
        
        if table_data is not None:
            processed_data['status_tables'].append(table_data)
        
        processed_data['notes'].append(note_table[1].css('td').css('pre::text').get())

        yield processed_data