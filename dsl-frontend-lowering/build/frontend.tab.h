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
    SET_UNWEIGHTED = 265,          /* SET_UNWEIGHTED  */
    SET_UNDIRECTED = 266,          /* SET_UNDIRECTED  */
    MODEL_W = 267,                 /* MODEL_W  */
    EVAL = 268,                    /* EVAL  */
    TRAIN = 269,                   /* TRAIN  */
    LAYER = 270,                   /* LAYER  */
    ITERS = 271,                   /* ITERS  */
    VAL_STEP = 272,                /* VAL_STEP  */
    AGGR_INIT = 273,               /* AGGR_INIT  */
    FN_ARG = 274,                  /* FN_ARG  */
    MUL_SUM = 275,                 /* MUL_SUM  */
    DSL_DOT = 276,                 /* DSL_DOT  */
    FFN_OUT = 277,                 /* FFN_OUT  */
    SIZE_FN = 278,                 /* SIZE_FN  */
    GRAPH_ATTR = 279,              /* GRAPH_ATTR  */
    FEAT_ATTR = 280,               /* FEAT_ATTR  */
    RELU = 281,                    /* RELU  */
    LABEL_ATTR = 282,              /* LABEL_ATTR  */
    DEGREE_ATTR = 283,             /* DEGREE_ATTR  */
    NODE_ATTR = 284,               /* NODE_ATTR  */
    LEAKY_RELU = 285,              /* LEAKY_RELU  */
    POW = 286,                     /* POW  */
    SCALAR_INIT = 287,             /* SCALAR_INIT  */
    COLTILE = 288,                 /* COLTILE  */
    FEAT_SIZE_ASSIGN = 289,        /* FEAT_SIZE_ASSIGN  */
    LABEL_SIZE_ASSIGN = 290,       /* LABEL_SIZE_ASSIGN  */
    COARSEN = 291,                 /* COARSEN  */
    SRC_ATTR = 292,                /* SRC_ATTR  */
    DST_ATTR = 293,                /* DST_ATTR  */
    INTEGER = 294,                 /* INTEGER  */
    FLOAT = 295,                   /* FLOAT  */
    SOFTMAX = 296,                 /* SOFTMAX  */
    INIT_WEIGHT = 297,             /* INIT_WEIGHT  */
    LBRACE = 298,                  /* LBRACE  */
    RBRACE = 299,                  /* RBRACE  */
    DOT = 300,                     /* DOT  */
    COMMA = 301,                   /* COMMA  */
    TR = 302,                      /* TR  */
    FA = 303,                      /* FA  */
    PLUS = 304,                    /* PLUS  */
    MINUS = 305,                   /* MINUS  */
    MULTIPLY = 306,                /* MULTIPLY  */
    DIVIDE = 307,                  /* DIVIDE  */
    FFN = 308,                     /* FFN  */
    NULL_KEY = 309,                /* NULL_KEY  */
    LOSS = 310,                    /* LOSS  */
    OPTIMIZER = 311,               /* OPTIMIZER  */
    RMSE_LOSS = 312,               /* RMSE_LOSS  */
    ADAM_T = 313,                  /* ADAM_T  */
    DSL_FN = 314,                  /* DSL_FN  */
    RELAXNLN = 315,                /* RELAXNLN  */
    QUANT = 316,                   /* QUANT  */
    RABBIT_REORDER_OP = 317,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 318,        /* SAMPLE_RANDOM_OP  */
    AGGR = 319,                    /* AGGR  */
    LSQBRA = 320,                  /* LSQBRA  */
    RSQBRA = 321,                  /* RSQBRA  */
    IF = 322,                      /* IF  */
    ELSE = 323,                    /* ELSE  */
    DO = 324,                      /* DO  */
    WHILE = 325,                   /* WHILE  */
    NOT = 326,                     /* NOT  */
    AND = 327,                     /* AND  */
    OR = 328,                      /* OR  */
    NOTEQ = 329,                   /* NOTEQ  */
    EQ = 330,                      /* EQ  */
    GREATER = 331,                 /* GREATER  */
    LESS = 332,                    /* LESS  */
    GREATEREQ = 333,               /* GREATEREQ  */
    LESSEQ = 334,                  /* LESSEQ  */
    DATASET = 335,                 /* DATASET  */
    NONLN = 336,                   /* NONLN  */
    SENSEI_OP = 337,               /* SENSEI_OP  */
    INT = 338,                     /* INT  */
    NEW = 339                      /* NEW  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 29 "frontend.y"

    int ival;
    float fval;
    char* sval;
    LayerOpType ltype;
    void* vval;

#line 156 "build/frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED  */
