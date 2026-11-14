"""Generate a standalone LaTeX report from the saved Python results.

Run after lab_sheet_3.py. All plots and tables are embedded in the .tex file;
compiling it requires no external images, data files, or bibliography files.
Compile with tectonic Lab_Sheet_III_LaTeX.tex or an equivalent LaTeX engine.
"""
from pathlib import Path
import csv
import json
import math

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
TARGET = HERE / 'Lab_Sheet_III_LaTeX.tex'
parts = []


def add(s):
    parts.append(s)


def read(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def num(value, digits=6):
    if value is None:
        return '--'
    if value == 0:
        return '$0$'
    mantissa, exponent = f'{float(value):.{digits-1}e}'.split('e')
    return rf'${mantissa}\times 10^{{{int(exponent)}}}$'


def method(name):
    return name.replace('Stormer', r'St\"ormer')


def tab(headers, rows, alignment=None, long=False):
    alignment = alignment or 'r' * len(headers)
    env = 'longtable' if long else 'tabular'
    add(r'\begin{center}\small' if not long else r'\begingroup\footnotesize\setlength{\tabcolsep}{5pt}')
    add(rf'\begin{{{env}}}{{{alignment}}}')
    heading = r'\toprule ' + ' & '.join(headers) + r' \\ \midrule'
    add(heading)
    if long:
        add(r'\endfirsthead')
        add(heading)
        add(r'\endhead\midrule\multicolumn{' + str(len(headers)) + r'}{r}{Continued on next page}\endfoot\bottomrule\endlastfoot')
    for row in rows:
        add(' & '.join(str(v) for v in row) + r' \\')
    if not long:
        add(r'\bottomrule')
    add(rf'\end{{{env}}}')
    add(r'\end{center}' if not long else r'\endgroup')


def convergence(rows):
    tab(['$h$', r'$\max_n|e_n|$', 'Error ratio', r'$\log_2$(ratio)'],
        [[f'{h:g}', num(e), '--' if ratio is None else f'{ratio:.6f}', '--' if p is None else f'{p:.6f}'] for h,e,ratio,p in rows])


def plot(curves, title, xlabel, ylabel, caption, kind='axis', extra='', height='6.5cm'):
    """Coordinates rather than images keep the .tex file self-contained."""
    add(r'\begin{figure}[H]\centering\begin{tikzpicture}')
    add(rf'\begin{{{kind}}}[width=0.94\linewidth,height={height},')
    add(rf'title={{{title}}},xlabel={{{xlabel}}},ylabel={{{ylabel}}},')
    add(r'grid=both,grid style={gray!20},tick label style={font=\small},')
    add(r'label style={font=\small},title style={font=\small},')
    add(r'legend style={font=\scriptsize,fill=white,draw=gray!40},legend pos=north east,' + extra + ']')
    colors = ['black', 'blue!75!black', 'orange!85!black', 'green!45!black', 'red!75!black']
    for i,(name,x,y,options) in enumerate(curves):
        coordinates = ' '.join(f'({float(a):.10g},{float(b):.12g})' for a,b in zip(x,y))
        add(rf'\addplot+[{colors[i % len(colors)]},no marks,line width=0.75pt,{options}] coordinates {{{coordinates}}};')
        add(rf'\addlegendentry{{{method(name)}}}')
    add(rf'\end{{{kind}}}\end{{tikzpicture}}\caption{{{caption}}}\end{{figure}}')


def xy(rows, ykey):
    return [float(r['x']) for r in rows], [float(r[ykey]) for r in rows]


def main():
    data = json.loads((OUT/'report_data.json').read_text())
    add(r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array}
\usepackage{xcolor,float,tikz,pgfplots}
\usepackage{fancyhdr}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\pgfplotsset{compat=1.18}
\definecolor{reportblue}{HTML}{183D65}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{3em}
\setcounter{tocdepth}{1}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small MTL 712 / MTL 7012}
\fancyhead[R]{\small Lab Sheet III}
\fancyfoot[C]{\small\thepage}
\newcommand{\St}{St\"ormer}
\newcommand{\err}{\mathcal E}
\title{\color{reportblue}\textbf{Lab Sheet III}\\[4pt]Numerical solutions in Python}
\author{MTL 712 / MTL 7012\\I Semester 2026--2027}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
\section*{Implementation and notation}
\addcontentsline{toc}{section}{Implementation and notation}
All numerical methods are implemented from scratch in Python. No built-in
ODE solver, root finder, numerical integration routine, or numerical
differentiation routine is used. NumPy supplies array arithmetic and
Matplotlib supplies the separate PNG plots. The figures in this report are
drawn by \texttt{pgfplots} from the same saved numerical data.

Let $x_n=x_0+nh$, $y_n\approx y(x_n)$, and $f_n=f(x_n,y_n)$.
The signed error and the maximum absolute error are
\[
e_n=y_n-y(x_n),\qquad \err(h)=\max_n|e_n|.
\]
The error ratio and observed convergence order are
\[
R(h)=\frac{\err(h)}{\err(h/2)},\qquad p_{\rm obs}=\log_2 R(h).
\]
Exact solutions are used only for comparison, never to initialize a recurrence.

\subsection*{RK4 starting values}
For the first-order system
\[
z=\begin{pmatrix}y\\v\end{pmatrix},\qquad
z'=F(x,z)=\begin{pmatrix}v\\f(x,y)\end{pmatrix},
\]
classical fourth-order Runge--Kutta is
\begin{align*}
K_1&=F(x_n,z_n),\\
K_2&=F(x_n+h/2,z_n+hK_1/2),\\
K_3&=F(x_n+h/2,z_n+hK_2/2),\\
K_4&=F(x_n+h,z_n+hK_3),\\
z_{n+1}&=z_n+\frac h6(K_1+2K_2+2K_3+K_4).
\end{align*}
Numerov and second-order \St{} use one RK4 step, third-order \St{} uses
two, and fourth-order \St{} uses three. Each RK4 step evaluates the scalar
acceleration $f(x,y)$ four times.

\subsection*{Reproducing the computation}
From the \texttt{tut3} directory, run
\begin{verbatim}
python -m pip install -r requirements.txt
python lab_sheet_3.py
python build_latex_report.py
tectonic Lab_Sheet_III_LaTeX.tex
python -m unittest discover -p "test_*.py"
\end{verbatim}
The generated \texttt{.tex} file is standalone: it embeds all plot coordinates
and table entries and needs no external images or data files.
The appendices contain all requested mesh tables. Full-precision data and
additional diagnostics are saved separately as CSV files.

\clearpage
\section{Numerov's method for a linear equation}
\[
y''=-y,\qquad y(0)=0,\quad y'(0)=1,\qquad 0\le x\le6,
\]
with exact solution $y(x)=\sin x$.

\subsection{Derivation of the explicit recurrence}
Add the Taylor expansions at $x_n+h$ and $x_n-h$:
\begin{equation}
y(x_n+h)-2y(x_n)+y(x_n-h)
=h^2y''(x_n)+\frac{h^4}{12}y^{(4)}(x_n)+O(h^6).
\label{eq:central-taylor}
\end{equation}
Writing $\phi(x)=f(x,y(x))=y''(x)$, so that $\phi''=y^{(4)}$, gives
\[
\phi(x_n+h)-2\phi(x_n)+\phi(x_n-h)=h^2\phi''(x_n)+O(h^4).
\]
Substituting into \eqref{eq:central-taylor}, combining the coefficients,
and omitting the $O(h^6)$ residual yields Numerov's formula:
\begin{equation}
\boxed{y_{n+1}-2y_n+y_{n-1}
=\frac{h^2}{12}\left(f_{n+1}+10f_n+f_{n-1}\right).}
\label{eq:numerov}
\end{equation}
For $f_j=-y_j$, this becomes
\[
\left(1+\frac{h^2}{12}\right)y_{n+1}
=\left(2-\frac{5h^2}{6}\right)y_n
-\left(1+\frac{h^2}{12}\right)y_{n-1}.
\]
Thus the explicit update is
\begin{equation}
\boxed{y_{n+1}=\frac{2-5h^2/6}{1+h^2/12}\,y_n-y_{n-1}.}
\end{equation}
The RK4 starter for this IVP is $y_1=h-h^3/6$. The residual of the
centered recurrence is $O(h^6)$, whereas the global solution error is
$O(h^4)$ with sufficiently accurate starting values.

\subsection{Measured accuracy}
''')
    convergence(data['q1'])
    add(r'The error ratios are approximately $16$, confirming global order $4$. '
        r'The sixth-order recurrence residual should not be confused with the global convergence order.')
    for h in [0.1,0.05,0.025]:
        rows=read(f'q1_h{h:g}.csv')
        x,y=xy(rows,'Numerov')
        exact=[float(r['exact']) for r in rows]
        plot([('Exact',x,exact,'dashed'),('Numerov',x,y,'')], rf'Linear Numerov, $h={h:g}$', '$x$', '$y$', rf'Numerov and $\sin x$ for $h={h:g}$. The two curves nearly coincide.')

    add(r'''\clearpage
\section{Numerov's method for a nonlinear equation}
\[
y''=6y^2,\qquad y(0)=1,\quad y'(0)=-2,\qquad 0\le x\le1,
\]
with exact solution $y(x)=(1+x)^{-2}$.

\subsection{From Numerov's formula to the nonlinear update}
The derivation of \eqref{eq:numerov} does not require $f$ to be linear.
Here $f_j=6y_j^2$. Substituting explicitly gives
\begin{align*}
y_{n+1}-2y_n+y_{n-1}
&=\frac{h^2}{12}\left(6y_{n+1}^2+60y_n^2+6y_{n-1}^2\right)\\
&=\frac{h^2}{2}\left(y_{n+1}^2+10y_n^2+y_{n-1}^2\right).
\end{align*}
Rearranging,
\begin{equation}
\boxed{y_{n+1}=2y_n-y_{n-1}
+\frac{h^2}{2}\left(y_{n+1}^2+10y_n^2+y_{n-1}^2\right).}
\end{equation}
At this step $y_n$ and $y_{n-1}$ are already known. Define their contribution as
\[
A_n=2y_n-y_{n-1}+\frac{h^2}{2}\left(10y_n^2+y_{n-1}^2\right).
\]
Then the equation for the next value is
\begin{equation}
\boxed{y_{n+1}=A_n+\frac{h^2}{2}y_{n+1}^2.}
\label{eq:nonlinear}
\end{equation}
This equation is implicit because its right-hand side contains the unknown
$y_{n+1}$, including its square.

\subsection{Fixed-point iteration}
Use a temporary guess $u^{(k)}$ for the value at the same next mesh point.
The initial guess is the explicit second-order \St{} predictor,
\[
u^{(0)}=2y_n-y_{n-1}+h^2f_n
=2y_n-y_{n-1}+6h^2y_n^2.
\]
Apply repeated corrections:
\begin{equation}
\boxed{u^{(k+1)}=A_n+\frac{h^2}{2}\left(u^{(k)}\right)^2,}
\end{equation}
stopping only when $|u^{(k+1)}-u^{(k)}|<10^{-12}$, and accept the latest
iterate as $y_{n+1}$. The subscript $n$ counts mesh steps; the superscript
$k$ counts guesses at one mesh point.

For $h=0.1$, $y_0=1$, and the RK4 starter $y_1=0.82647225$,
\[
A_1\approx0.692097319,\qquad
u^{(k+1)}\approx0.692097319+0.005\left(u^{(k)}\right)^2.
\]
This iteration yields $y_2\approx0.694509033$.
The derivative of the iteration map is $g'(u)=h^2u$, which is small near
the physical solution branch. Its magnitude is approximately at most $0.01$
for $h=0.1$ here, explaining the rapid convergence.
The implementation caps the iteration count and rejects non-finite values;
it does not use the quadratic formula in place of the required iteration.

\subsection{Accuracy and iteration counts}
''')
    tab(['$h$', r'$\max_n|e_n|$', 'Min.', 'Max.', 'Mean iterations'],
        [[f'{h:g}',num(e),lo,hi,f'{avg:.3f}'] for h,e,lo,hi,avg in data['q2']])
    ratio=data['q2'][0][1]/data['q2'][1][1]
    add(rf'Halving $h$ reduces the maximum error by ${ratio:.4f}$, giving '
        rf'$p_{{\rm obs}}={math.log2(ratio):.4f}$. The coarse-grid ratio is approaching '
        r'fourth-order behavior; RK4 startup error and higher-order terms influence '
        r'the result at these step sizes. The mean iteration count decreases '
        r'because the contraction factor scales with $h^2$.')
    add(r'Iteration counts are zero at $x=0$ (prescribed data) and $x=h$ '
        r'(RK4 starter). Subsequent counts record the number of corrections, '
        r'excluding the predictor. A small change between iterates controls the '
        r'implicit equation solve, not the discretization or startup errors.')
    for h in [0.1,0.05]:
        rows=read(f'q2_h{h:g}.csv')
        x,y=xy(rows,'Numerov')
        plot([('Exact',x,[float(r['exact']) for r in rows],'dashed'),('Numerov',x,y,'')],
             rf'Nonlinear Numerov, $h={h:g}$', '$x$', '$y$', rf'Numerov and $(1+x)^{{-2}}$ at $h={h:g}$.')

    add(r'''\clearpage
\section{Second-, third-, and fourth-order St\"ormer formulas}
The sine IVP of Section 1 is solved at $h=0.1$, $0.05$, $0.025$, and $0.0125$.

\subsection{Backward-difference derivation}
Define $\nabla f_n=f_n-f_{n-1}$. The exact centered difference has the
integral representation
\[
y(x_n+h)-2y(x_n)+y(x_n-h)
=h^2\int_{-1}^{1}(1-|s|)\phi(x_n+sh)\,ds,
\]
where $\phi(x)=f(x,y(x))$. Expand by Newton's backward interpolation series:
\[
\phi(x_n+sh)=\phi_n+s\nabla\phi_n
+\frac{s(s+1)}{2}\nabla^2\phi_n
+\frac{s(s+1)(s+2)}{6}\nabla^3\phi_n+\cdots.
\]
Integrating the coefficients against $1-|s|$ gives $1$, $0$, $1/12$,
and $1/12$, respectively. For example,
\[
\int_{-1}^{1}(1-|s|)\frac{s(s+1)}2\,ds
=\int_0^1(1-s)s^2\,ds=\frac1{12}.
\]
Thus
\[
y_{n+1}-2y_n+y_{n-1}
=h^2\left(f_n+\frac1{12}\nabla^2f_n+\frac1{12}\nabla^3f_n+\cdots\right).
\]
The first-difference coefficient is zero. Therefore the sheet's first,
second, and third terms are interpreted as the first three \emph{nonzero}
terms, yielding the requested orders $2$, $3$, and $4$:
\begin{align}
\mathrm{S2}:\quad y_{n+1}&=2y_n-y_{n-1}+h^2f_n,\\
\mathrm{S3}:\quad y_{n+1}&=2y_n-y_{n-1}
+\frac{h^2}{12}(13f_n-2f_{n-1}+f_{n-2}),\\
\mathrm{S4}:\quad y_{n+1}&=2y_n-y_{n-1}
+\frac{h^2}{12}(14f_n-5f_{n-1}+4f_{n-2}-f_{n-3}).
\end{align}
The recurrence residuals are $O(h^4)$, $O(h^5)$, and $O(h^6)$, giving
global orders $2$, $3$, and $4$. In particular,
$\nabla^2\phi_n=h^2\phi''(x_n)-h^3\phi^{(3)}(x_n)+O(h^4)$
and $\nabla^3\phi_n=h^3\phi^{(3)}(x_n)+O(h^4)$, so the third-difference
term cancels the leading $h^5$ residual left by S3.

RK4 supplies the required starting values, and each previously computed
acceleration is cached. The general backward-difference representation is
also given by Grazier et al.~\cite{grazier}, equations (2)--(3).

\subsection{Measured convergence}
''')
    for order in [2,3,4]:
        add(rf'\subsubsection*{{Order {order}}}')
        convergence(data['q3'][str(order)])
    add(r'The error ratios tend to $4$, $8$, and $16$, confirming orders '
        r'$2$, $3$, and $4$. The corresponding log--log slopes agree with these orders.')
    curves=[]
    for order in [2,3,4]:
        rows=data['q3'][str(order)]
        curves.append((f'Stormer {order}',[r[0] for r in rows],[r[1] for r in rows],'mark=*'))
    plot(curves, r'\St{} convergence', '$h$', r'$\max_n|e_n|$',
         r'Maximum absolute error against step size on logarithmic axes.', kind='loglogaxis')

    add(r'''\clearpage
\section{Nonlinear pendulum and energy}
\[
y''=-\sin y,\qquad y(0)=1,\quad y'(0)=0,\qquad0\le x\le50.
\]
S2, S4, and RK4 use $h=0.05$.
\subsection{Energy and the velocity approximation}
The exact energy is
\[
E(x)=\frac12 v(x)^2-\cos y(x).
\]
Indeed,
\[
E'(x)=vv'+\sin(y)y'=v(-\sin y)+\sin(y)v=0,
\]
so $E_0=-\cos1=-0.5403023058681398$.
At mesh points use the prescribed central difference
\[
v_n^{\rm cd}=\frac{y_{n+1}-y_{n-1}}{2h},\qquad
E_n^{\rm cd}=\frac12(v_n^{\rm cd})^2-\cos y_n.
\]
At $x=0$ the given velocity is used. Each method takes one extra step to
$x=50.05$ to permit a central difference at $x=50$. This extra point is
excluded from every reported interval and mesh table.
RK4 is postprocessed by the same central difference for the main comparison,
while its directly integrated velocity is retained for an additional diagnostic.
''')
    tab(['Method', r'$\max|E_n^{\rm cd}-E_0|$', r'$|E^{\rm cd}(50)-E_0|$', 'Native RK4 max.'],
        [[method(m),num(a),num(b),num(c)] for m,a,b,c in data['q4']], alignment='lrrr')
    add(r'S2 has bounded, oscillatory energy error. S4 exhibits a small accumulating '
        r'drift superposed on oscillation. RK4 conserves its native energy very '
        r'accurately over this interval. Central differences introduce $O(h^2)$ '
        r'velocity error, which dominates the S4 and RK4 central-energy diagnostics. '
        r'The smaller central-energy error of S2 can therefore result from cancellation '
        r'and does not establish a more accurate trajectory.')
    rk=read('q4_RK4.csv')
    comparisons=[]
    pendulum={}
    for name in ['Stormer 2','Stormer 4','RK4']:
        rows=read('q4_'+name.replace(' ','_')+'.csv')
        pendulum[name]=rows
        if name != 'RK4':
            difference=max(abs(float(a['y'])-float(b['y'])) for a,b in zip(rows,rk))
            comparisons.append([method(name),num(difference)])
    tab(['Trajectory compared with RK4', r'$\max_n|y_n-y_n^{\rm RK4}|$'], comparisons, alignment='lr')
    add(r'These values measure discrepancies between numerical trajectories, '
        r'not errors against an exact pendulum solution. S4 agrees much more '
        r'closely with RK4 than S2 does.')
    curves=[]
    for name,rows in pendulum.items():
        x,y=xy(rows,'y'); curves.append((name,x,y,''))
    plot(curves,'Pendulum trajectories','$x$','$y$ (radians)',r'Pendulum displacement on $[0,50]$, $h=0.05$.')
    curves=[]
    for name,rows in pendulum.items():
        curves.append((name,[float(r['y']) for r in rows],[float(r['central_velocity']) for r in rows],''))
    plot(curves,'Pendulum phase plane','$y$ (radians)',r'$v^{\rm cd}$',r'Phase trajectories using central-difference velocities.',extra='axis equal image,')
    curves=[]
    for name,rows in pendulum.items():
        x,y=xy(rows,'abs_energy_error');curves.append((name,x,y,''))
    plot(curves,'Central-difference energy error','$x$',r'$|E_n^{\rm cd}-E_0|$',r'Energy errors computed consistently from central-difference velocities.')
    rows=rk[1:]
    x=[float(r['x']) for r in rows]
    plot([('Central velocity',x,[max(float(r['abs_energy_error']),1e-18) for r in rows],''),
          ('Native RK4 velocity',x,[max(float(r['RK4_native_abs_energy_error']),1e-18) for r in rows],'')],
         'RK4 velocity postprocessing','$x$',r'$|E_n-E_0|$',
         r'RK4 native energy reveals the smaller integration error masked by central differences.',kind='semilogyaxis')

    add(r'''\clearpage
\section{Comparative study on \texorpdfstring{$[0,20]$}{[0,20]}}
\[
y''=-y,\qquad y(0)=1,\quad y'(0)=0,\qquad y(x)=\cos x.
\]
Numerov, S2, S4, and RK4 are compared at $h=0.1$ and $h=0.05$.
\subsection{Accuracy and acceleration evaluations}
''')
    tab(['$h$','Method',r'$\max_n|e_n|$','Evaluations of $f$'],
        [[h,method(m),num(e),c] for h,m,e,c in data['q5']], alignment='rlrr')
    add(r'At both prescribed step sizes the accuracy ranking is Numerov, RK4, '
        r'S4, then S2. The fourth-order methods reduce their maximum errors by '
        r'approximately $16$ when $h$ is halved; S2 reduces its error by approximately $4$.')
    add(r'''Let $N=20/h$. The actual scalar-acceleration call counts, including startup, are
\begin{align*}
C_{\rm Numerov}&=4+N,\\
C_{\rm S2}&=4+(N-1)=N+3,\\
C_{\rm S4}&=12+N,\\
C_{\rm RK4}&=4N.
\end{align*}
Numerov caches $f_0,\ldots,f_{N-1}$; the unknown $f_{n+1}=-y_{n+1}$
is incorporated algebraically into the denominator. S2 only uses
$f_1,\ldots,f_{N-1}$, and S4 uses $f_0,\ldots,f_{N-1}$.
No unnecessary acceleration at the final mesh point is counted.

For this linear equation, replacing all known force calls by their algebraic
values $-y_n$ could further reduce call counts. The recorded values retain
the actual known-value calls in the cached-force implementation; they do
not claim to measure optimized wall-clock runtime or all arithmetic work.
''')
    for h in [0.1,0.05]:
        rows=read(f'q5_h{h:g}.csv')
        x=[float(r['x']) for r in rows]
        curves=[('Exact',x,[float(r['exact']) for r in rows],'dashed')]
        for name in ['Numerov','Stormer 2','Stormer 4','RK4']:
            curves.append((name,x,[float(r[name]) for r in rows],''))
        plot(curves,rf'Solution comparison, $h={h:g}$','$x$','$y$',rf'All four numerical methods and $\cos x$ at $h={h:g}$.')
        curves=[]
        for name in ['Numerov','Stormer 2','Stormer 4','RK4']:
            curves.append((name,x[1:],[max(float(r[name+'_abs_error']),1e-18) for r in rows[1:]],''))
        plot(curves,rf'Absolute errors, $h={h:g}$','$x$',r'$|e_n|$',
             rf'Semilogarithmic error comparison at $h={h:g}$. The initial point is omitted; exact zero errors are floored at $10^{{-18}}$ only for plotting.',kind='semilogyaxis')
    add(r'''\subsection{Accuracy at equal computational work}
For each acceleration-call budget, use the largest integer step count
permitted by the formulas above. Each method therefore uses its own step
size $h=20/N$ while solving on the same interval.
''')
    tab(['Budget','Method','$N$','$h$','Calls used',r'$\max_n|e_n|$'],
        [[b,method(m),n,f'{h:.6f}',c,num(e)] for b,m,n,h,c,e in data['work'] if b in [200,800]],alignment='rlrrrr')
    curves=[]
    for name in ['Numerov','Stormer 2','Stormer 4','RK4']:
        rows=[r for r in data['work'] if r[1]==name]
        curves.append((name,[r[4] for r in rows],[r[5] for r in rows],'mark=*'))
    plot(curves,'Accuracy at equal force-call budgets','Evaluations of $f$',r'$\max_n|e_n|$',
         r'Maximum error versus actual acceleration-call count, including RK4 starting values.',kind='loglogaxis')
    add(r'Numerov has the smallest maximum error at all tested budgets '
        r'$100$, $200$, $400$, $800$, and $1600$. It combines fourth-order '
        r'accuracy, a small error constant on this harmonic IVP, and roughly '
        r'one acceleration evaluation per step. S4 also requires roughly one '
        r'evaluation per step, but its error constant is larger here. RK4 '
        r'has four evaluations per step, and S2 has only second-order accuracy. '
        r'These conclusions concern the equations, intervals, and work measure tested.')

    add(r'''\clearpage
\begin{thebibliography}{1}
\bibitem{grazier} K. R. Grazier, W. I. Newman, J. M. Hyman,
P. W. Sharp, and D. J. Goldstein,
\emph{Achieving Brouwer's law with high-order St\"ormer multistep methods},
ANZIAM Journal 46(E), C786--C804, 2005.
\url{https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/viewFile/990/865}.
\end{thebibliography}
\appendix
\clearpage
\section{Complete linear Numerov tables}
Signed error means numerical minus exact. All mesh points are included;
rounding applies only to presentation.
''')
    for h in [0.1,0.05,0.025]:
        add(r'\subsection{\texorpdfstring{$h=' + f'{h:g}' + r'$}{h=' + f'{h:g}' + r'}}')
        rows=read(f'q1_h{h:g}.csv')
        tab(['$x_n$','$y_n$','$\sin x_n$','$e_n$','$|e_n|$'],
            [[f'{float(r["x"]):.3f}',f'{float(r["Numerov"]):.10f}',f'{float(r["exact"]):.10f}',num(float(r['signed_error']),5),num(float(r['absolute_error']),5)] for r in rows],long=True)
        add(r'\clearpage')
    add(r'\section{Complete nonlinear Numerov tables}')
    for h in [0.1,0.05]:
        add(r'\subsection{\texorpdfstring{$h=' + f'{h:g}' + r'$}{h=' + f'{h:g}' + r'}}')
        rows=read(f'q2_h{h:g}.csv')
        tab(['$x_n$','$y_n$','Exact $y$','Iterations','$e_n$','$|e_n|$'],
            [[f'{float(r["x"]):.2f}',f'{float(r["Numerov"]):.10f}',f'{float(r["exact"]):.10f}',r['iterations'],num(float(r['signed_error']),5),num(float(r['absolute_error']),5)] for r in rows],long=True)
    add(r'''\clearpage\section{Complete four-method mesh tables}
All requested mesh points and all four numerical approximations are included.
Separate CSV files also provide each method's absolute errors.
''')
    for h in [0.1,0.05]:
        add(r'\subsection{\texorpdfstring{$h=' + f'{h:g}' + r'$}{h=' + f'{h:g}' + r'}}')
        rows=read(f'q5_h{h:g}.csv')
        tab(['$x_n$','$\cos x_n$','Numerov',r'\St{} 2',r'\St{} 4','RK4'],
            [[f'{float(r["x"]):.2f}']+[f'{float(r[k]):.9f}' for k in ['exact','Numerov','Stormer 2','Stormer 4','RK4']] for r in rows],long=True)
        if h == 0.1:
            add(r'\clearpage')
    add(r'\end{document}')
    TARGET.write_text('\n'.join(parts))
    print('Created standalone source:',TARGET)


if __name__ == '__main__':
    main()
