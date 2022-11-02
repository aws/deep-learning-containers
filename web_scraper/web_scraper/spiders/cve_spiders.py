import scrapy
import os


class CveSpider(scrapy.Spider):
    name = "cve" ## Spider name should be unique

    ## Feed Settings required to get a beautifully arranged json format
    custom_settings = {'FEEDS':{
                                'scraped_data.json': {
                                    'format': 'json',
                                    'encoding': 'utf8',
                                    'indent': 4,
                                }
                            }
                      }
    
    def __init__(self, url_csv_string=None, *args, **kwargs):
        super(CveSpider, self).__init__(*args, **kwargs)
        ## For the data to be stored properly, make sure that the URLs are unique. 
        self.urls = list(set(url_csv_string.split(',')))

    def start_requests(self):
        ## Parse function stores the data related to each CVE in a JSON format.
        for url in self.urls:
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
            row_data_list = row.css('td::text').getall()
            if len(row.css('td')) >= 1:
                row_data_list = row_data_list + row.css('td')[-1].css('small::text').getall()
            processed_tuple = self.process_row(row_data_list)
            if processed_tuple[0] is None:
                continue
            if len(row.css('tr').css('td')) == 3:
                ## If there are 3 td (table data) sections inside a table row, then each section stands for a 
                ## column. In other words, when we get a row with 3 columns, we know that the first column
                ## is for the package name.
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
        
        all_notes = []
        if len(note_table) > 0:
            all_notes = note_table[1].css('td').css('pre::text').getall()
        processed_data['notes'] = [note for note in all_notes]

        yield processed_data