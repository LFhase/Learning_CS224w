# VS Code Setting for Python

Share my settings of vscode for python coding for research purpose

### Editor Setting

Extensions you need to download:
- Python: Great extension producted by MS
- PyLint: Check for grammar mistakes or coding regulations 
- Yapf: Powerful formatter
- Autodocstring: Powerful doc helper and I have a personal template here `lfhase.mustache`
- LaTeX Workshop: Help with LaTeX writing. Of course you also need a TexLive.

Here is my user setting which would exist in `workspace\.vscode\settings.json`
```
{
    "editor.rulers": [
        120
    ],
    "python.formatting.yapfPath": "D:\\python\\Scripts\\yapf.exe",
    "python.linting.pylintEnabled": true,
    "python.linting.pylintArgs": [
        "--generated-members=numpy.* ,torch.*",
        "--max-line-length=120"
    ],
    "python.formatting.provider": "yapf",
    "editor.formatOnSave": true,
    "python.formatting.yapfArgs": [
        "--style={ based_on_style: google, column_limit: 120 }"
    ],
}
```

### ShortCuts
`Ctrl + D`: Copy line down <br>
`Ctrl + \``: Open a new teminal