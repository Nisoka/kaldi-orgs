TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += "/nwork/svn/ai/sr/kaldi/kaldi-orgs/rnn"

SOURCES += main.cpp \
    ../rnn_binary.cc

HEADERS += \
    ../rnn_binary.h
