/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED
# define YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    IDENTIFIER = 258,              /* IDENTIFIER  */
    ASSIGN = 259,                  /* ASSIGN  */
    LOAD = 260,                    /* LOAD  */
    LPAREN = 261,                  /* LPAREN  */
    RPAREN = 262,                  /* RPAREN  */
    SEMICOLON = 263,               /* SEMICOLON  */
    QUOTE = 264,                   /* QUOTE  */
    COMMENT = 265,                 /* COMMENT  */
    SET_UNWEIGHTED = 266,          /* SET_UNWEIGHTED  */
    SET_UNDIRECTED = 267,          /* SET_UNDIRECTED  */
    MODEL_W = 268,                 /* MODEL_W  */
    EVAL = 269,                    /* EVAL  */
    TRAIN = 270,                   /* TRAIN  */
    LAYER = 271,                   /* LAYER  */
    LOSS = 272,                    /* LOSS  */
    OPTIMIZER = 273,               /* OPTIMIZER  */
    ITERS = 274,                   /* ITERS  */
    VAL_STEP = 275,                /* VAL_STEP  */
    RMSE_LOSS = 276,               /* RMSE_LOSS  */
    ADAM_T = 277,                  /* ADAM_T  */
    AGGR_INIT = 278,               /* AGGR_INIT  */
    FN_ARG = 279,                  /* FN_ARG  */
    MUL_SUM = 280,                 /* MUL_SUM  */
    DSL_FN = 281,                  /* DSL_FN  */
    DSL_DOT = 282,                 /* DSL_DOT  */
    FFN_OUT = 283,                 /* FFN_OUT  */
    SIZE_FN = 284,                 /* SIZE_FN  */
    RELAXNLN = 285,                /* RELAXNLN  */
    QUANT = 286,                   /* QUANT  */
    GRAPH_ATTR = 287,              /* GRAPH_ATTR  */
    FEAT_ATTR = 288,               /* FEAT_ATTR  */
    RELU = 289,                    /* RELU  */
    LABEL_ATTR = 290,              /* LABEL_ATTR  */
    RABBIT_REORDER_OP = 291,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 292,        /* SAMPLE_RANDOM_OP  */
    COLTILE = 293,                 /* COLTILE  */
    AGGR = 294,                    /* AGGR  */
    FEAT_SIZE_ASSIGN = 295,        /* FEAT_SIZE_ASSIGN  */
    LABEL_SIZE_ASSIGN = 296,       /* LABEL_SIZE_ASSIGN  */
    COARSEN = 297,                 /* COARSEN  */
    INTEGER = 298,                 /* INTEGER  */
    FLOAT = 299,                   /* FLOAT  */
    LBRACE = 300,                  /* LBRACE  */
    RBRACE = 301,                  /* RBRACE  */
    LSQBRA = 302,                  /* LSQBRA  */
    RSQBRA = 303,                  /* RSQBRA  */
    DOT = 304,                     /* DOT  */
    COMMA = 305,                   /* COMMA  */
    IF = 306,                      /* IF  */
    ELSE = 307,                    /* ELSE  */
    DO = 308,                      /* DO  */
    WHILE = 309,                   /* WHILE  */
    TR = 310,                      /* TR  */
    FA = 311,                      /* FA  */
    NOT = 312,                     /* NOT  */
    AND = 313,                     /* AND  */
    OR = 314,                      /* OR  */
    NOTEQ = 315,                   /* NOTEQ  */
    EQ = 316,                      /* EQ  */
    GREATER = 317,                 /* GREATER  */
    LESS = 318,                    /* LESS  */
    GREATEREQ = 319,               /* GREATEREQ  */
    LESSEQ = 320,                  /* LESSEQ  */
    PLUS = 321,                    /* PLUS  */
    MINUS = 322,                   /* MINUS  */
    MULTIPLY = 323,                /* MULTIPLY  */
    DIVIDE = 324,                  /* DIVIDE  */
    FFN = 325,                     /* FFN  */
    DATASET = 326,                 /* DATASET  */
    NONLN = 327,                   /* NONLN  */
    SENSEI_OP = 328,               /* SENSEI_OP  */
    INT = 329,                     /* INT  */
    NEW = 330,                     /* NEW  */
    NULL_KEY = 331                 /* NULL_KEY  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 25 "frontend.y"

    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
    ForwardNode* forwardNode;
    TrainingLoopNode* trainingLoopNode;

#line 149 "build/frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED  */
