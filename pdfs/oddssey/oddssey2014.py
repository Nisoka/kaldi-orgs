#!/usr/bin/env python

# Copyright
#http://www.speakerodyssey.com/


from bs4 import BeautifulSoup
import urllib
import ssl
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

DATA_PATH = "/opt/doc"
DATA_PATH = "D://workspace//weekMeeting//doc"
ODD_VER = '2014'
htmlRoot = 'https://www.isca-speech.org/archive/odyssey_2014/'
#view-source:https://www.isca-speech.org/archive/odyssey_2014/pdfs/22.pdf


import os, errno 
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def Schedule(a,b,c):
  per = 100.0 * a * b / c
  if per > 100 :
    per = 100
  print '%.2f%%' % per


def getInnerHtml(html ,ref):
  ihtml = html[:html.rindex('/')+1]
  return ihtml + ref

def getRealPaperPath(html, title, filepath, suffix):
  context = ssl._create_unverified_context()
  page = urllib.urlopen(html,context=context)
  soup = BeautifulSoup(page , 'html5lib')
  docPath = htmlRoot + "pdfs/" + str(suffix) + ".pdf"
  urllib.urlretrieve(docPath, os.path.join(filepath, title) , Schedule, context=context)





context = ssl._create_unverified_context()
page = urllib.urlopen(htmlRoot,context=context)
soup = BeautifulSoup(page , 'html5lib')
kps = soup.select('td.td_session')
for i,kp in enumerate(kps):
  #print kp
  print "i=%s" % i
  #print kp.parent.next_sibling.next_sibling
  head = kp.get_text()
  print head
  topicPapers = kp.parent.next_sibling.next_sibling.select('td.td_paper')
  subdir = os.path.join(DATA_PATH, ODD_VER)
  head = head.strip()
  topicdir = os.path.join(subdir, head)
  mkdir_p(topicdir)
  
  for ps in topicPapers:
    ahref = ps.a
    print ahref.contents[0]
    gp = re.search('\d+', ahref['href'])
    print ahref['href']
    if gp is not None:
	  print gp.group(0)
	  pdfName = ahref.contents[0]
	  pdfName = pdfName.replace(':','=')
	  pdfName = pdfName.replace('?','')
	  pdfName = pdfName.replace('/','-')
	  pdfName = pdfName.strip()
	  getRealPaperPath(htmlRoot, pdfName + ".pdf", topicdir, gp.group(0))
    else:
      print "is none"
  

  


#print soup.prettify()
#print soup
