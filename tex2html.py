import re

filename = '/Users/tillhoffmann/Dropbox/Notes/Graphical model and Gibbs/graphical_gibbs.tex'

fp = open(filename)
text = fp.read()
fp.close()

#Get the main document
document = re.search(r"\\begin{document}(.*?)\\end{document}", text, re.DOTALL).group(1)

#Replace all inline dollar characters for equations
document = re.sub(r"([^\\])\$(.*?[^\\])\$", r"\1\\(\2\\)", document)

#Replace all single command formats
for tex, html in {'emph': 'strong',
                  'section': 'h2',
                  'subsection': 'h3',
                  'subsubsection': 'h4'}.iteritems():
    document = re.sub(r"\\%s{(.*?)}" % tex, r"<%s>\1</%s>" % (html, html), document)

#Create a bibliography by finding all cite commands
matches = re.findall(r"\\cite{(.*?)}", document)
reference_counter = 0
references = {}
#Go over all citations
for citekey in matches:
    if citekey not in references:
        reference_counter += 1
        references[citekey] = reference_counter
        #Replace all occurrences
        document = re.sub(r"\\cite{%s}" % citekey, "[{}]".format(reference_counter), document)

#Append the bibliography
if len(references) > 0:
    references = "\n".join(["<li>{}</li>".format(key) for key, _ in sorted(references.iteritems(), key=lambda x:x[1])])
    document += """<h2>References</h2>

<ol>
{}
</ol>""".format(references)

#Process all the figures
figure_counter = 0
figures = {}

def sub_figure(match):
    global figures, figure_counter
    text = match.group(1)
    #Get all the graphics
    graphics = re.findall(r"\\includegraphics(?:\[.*?\])?{(.*?)}", text)

    #Get the caption
    caption = re.search(r"\\caption", text)
    assert caption is not None, "Missing caption."
    caption = text[caption.end():]
    bracket_count = 0
    for i in range(len(caption)):
        if caption[i] == '{':
            bracket_count += 1
        elif caption[i] == '}':
            bracket_count -= 1
        if bracket_count == 0:
            end = i
            break
    #Replace the label tag
    caption = re.sub(r"\\label{.*?}", "", caption[1:end])

    #Get the label
    label = re.search(r"\\label{(.*?)}", text)
    if label is None:
        label = reference_counter + 1
    else:
        label = label.group(1)

    if label not in figures:
        figure_counter += 1
        figures[label] = figure_counter

    figure_ref = figures[label]

    graphics = ['<img src="%s.png"/>' % graphic for graphic in graphics]

    return '''<p>
    <div class="figure">
        <center>{imgs}</center>
        <div><strong>Figure {ref}:</strong>{caption}</div>
    </div>
</p>'''.format(caption=caption, ref=figure_ref, imgs="".join(graphics))

document = re.sub(r"\\begin{figure\*?}(.*?)\\end{figure\*?}", sub_figure, document, flags=re.DOTALL)

#Replace all the figure references
for label, ref in figures.iteritems():
    document = re.sub(r"\\ref{%s}" % label, str(ref), document)

document = re.sub(r"~", r"&nbsp;", document)

#Replace all backslash by their unicode character
document = re.sub(r"\\", r"&#92;", document)
print document