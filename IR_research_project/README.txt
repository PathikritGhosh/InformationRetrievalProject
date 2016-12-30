For classic dataset -

1. In the current working directory, extract the dataset to a folder "classic" with no sub-directories (all documents placed in classic directory itself)
2. Create a folder "Classic_Dataset" with empty sub-directories "cacm", "cisi", "cran", "med" by executing -
"mkdir -p Classic_Dataset/{cacm,cisi,cran,med}"
3. execute "python ir_project_classic.py x y" where top x% features are extracted using variance-based feature selection and from those top y% are selected using idf values

For WebKB dataset -

1. In the current working directory, extract the dataset to a folder "WebKB" with all documents placed in their respective sub-directories
2. Create a folder "WebKB_Dataset" with empty sub-directories "course", "faculty", "project", "student"
"mkdir -p WebKB_Dataset/{course,faculty,project,student}"
3. execute "python ir_project_webkb.py x y d" where top x% features are extracted using variance-based feature selection and from those top y% are selected using idf values. d is the number of documents to be taken from each class
