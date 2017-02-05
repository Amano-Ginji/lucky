#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import urllib2

from HTMLParser import HTMLParser

# data sources: 
# url = "http://www.baidu.com"
# url = "http://caipiao.163.com/award/cqssc/20170101.html"
# url = "http://baidu.lecai.com/lottery/draw/list/200?d=2017-01-01"

headers = { #伪装为浏览器抓取  
    'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'  
}  

url = sys.argv[1]
req = urllib2.Request(url, headers=headers)
content = urllib2.urlopen(req).read()

class DataParser(HTMLParser):
    selected = ['table', 'tbody', 'tr', 'td']
    selected_sid = "table/tbody/tr/td"

    def __init__(self):
        HTMLParser.__init__(self)
        self.last_class = ''
        self.sid = ''
        self.dict_sid_result = {}

    def reset(self):
        HTMLParser.reset(self)
        self._level_stack = []
        self.last_class = ''
        self.sid = ''

    def get_attr(self, attrs, name):
        for attr in attrs:
            if attr[0] == name:
                return attr[1]
        return None

    def handle_starttag(self, tag, attrs):  
        if tag in DataParser.selected:
            self._level_stack.append(tag)  
            self.last_class = self.get_attr(attrs, 'class')
    
    def handle_endtag(self, tag):  
        if self._level_stack and tag in DataParser.selected and tag == self._level_stack[-1]:
            self._level_stack.pop()  
    
    def handle_data(self, data):
        if "/".join(self._level_stack) == DataParser.selected_sid:
            data = data.strip()
            if data and self.last_class == 'td2':
                self.sid = data
            if data and self.last_class == 'td3':
                if self.sid.strip():
                    if self.sid not in self.dict_sid_result:
                        self.dict_sid_result[self.sid] = []
                    self.dict_sid_result[self.sid].append(data)

dp = DataParser()
dp.feed(content)

for k in sorted(dp.dict_sid_result.keys()):
    print("%s\t%s" % (k, ",".join(dp.dict_sid_result[k])))



