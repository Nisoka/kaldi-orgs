#!/usr/bin/env python

# Copyright


from bs4 import BeautifulSoup
import urllib
import ssl
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

DATA_PATH = "/opt/doc"
DATA_PATH = "D://workspace//weekMeeting//doc"
ODD_VER = '2004'
htmlRoot = 'https://www.isca-speech.org/archive_open/odyssey_04/index.html'
#htmlRoot = 'https://www.isca-speech.org/archive_open/odyssey_2008/index.html'
#htmlRoot = 'https://www.isca-speech.org/archive_open/odyssey_2010/index.html'
#htmlRoot = 'https://www.isca-speech.org/archive/odyssey_2012/index.html'





import os
def Schedule(a,b,c):
  per = 100.0 * a * b / c
  if per > 100 :
    per = 100
  print '%.2f%%' % per


def getInnerHtml(html ,ref):
  ihtml = html[:html.rindex('/')+1]
  return ihtml + ref

def getRealPaperPath(html, title, filepath):
  context = ssl._create_unverified_context()
  page = urllib.urlopen(html,context=context)
  soup = BeautifulSoup(page , 'html5lib')
  lens = len(soup.find_all('p'))
  print "lens=%s" % ( lens)
  if lens < 3:
    return
  info = soup.find_all('p')[lens-3].get_text()
  if soup.select('p')[lens-2].select('a') == None or len(soup.select('p')[lens-2].select('a')) == 0:
    return
  realPath = soup.select('p')[lens-2].select('a')[0]['href']
  docPath = getInnerHtml(html, realPath)
  print docPath
  urllib.urlretrieve(docPath, os.path.join(filepath, title) , Schedule, context=context)


import os, errno 
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise



context = ssl._create_unverified_context()
page = urllib.urlopen(htmlRoot,context=context)
soup = BeautifulSoup(page , 'html.parser')
print soup.select('h3')
kps = soup.select('h3')[1]

#all papers
allPapers = kps.find_all_next('a')
for i, paper in enumerate(allPapers):
 
  dirName = paper.find_previous('h3')
  if paper.has_attr('href'):
    pass
  else:
    print 'has no attr============================================nnnn'
    continue
  print dirName.get_text()
  print paper
  subdir = os.path.join(DATA_PATH, ODD_VER)
  topicdir = os.path.join(subdir, dirName.get_text())
  mkdir_p(topicdir)

  title = re.search( '\"(.*?)\"', paper.get_text())
  if title == None:
    continue
  else:
    title = title.group(0)
  print type(title)
  title = title.replace('\"', '')
  if title == "":
    title = dirName.get_text() + str(i)
  print "  %s" % (title)
  title = title + ".pdf"
  title = title.replace('/','-')
  getRealPaperPath( getInnerHtml(htmlRoot, paper['href']), title, topicdir)
  


#print soup.prettify()
#print soup
