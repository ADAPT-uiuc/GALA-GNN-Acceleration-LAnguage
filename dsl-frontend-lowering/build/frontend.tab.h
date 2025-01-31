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
    DEGREE_ATTR = 291,             /* DEGREE_ATTR  */
    NODE_ATTR = 292,               /* NODE_ATTR  */
    LEAKY_RELU = 293,              /* LEAKY_RELU  */
    RABBIT_REORDER_OP = 294,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 295,        /* SAMPLE_RANDOM_OP  */
    POW = 296,                     /* POW  */
    SCALAR_INIT = 297,             /* SCALAR_INIT  */
    COLTILE = 298,                 /* COLTILE  */
    AGGR = 299,                    /* AGGR  */
    FEAT_SIZE_ASSIGN = 300,        /* FEAT_SIZE_ASSIGN  */
    LABEL_SIZE_ASSIGN = 301,       /* LABEL_SIZE_ASSIGN  */
    COARSEN = 302,                 /* COARSEN  */
    SRC_ATTR = 303,                /* SRC_ATTR  */
    DST_ATTR = 304,                /* DST_ATTR  */
    INTEGER = 305,                 /* INTEGER  */
    FLOAT = 306,                   /* FLOAT  */
    SOFTMAX = 307,                 /* SOFTMAX  */
    INIT_WEIGHT = 308,             /* INIT_WEIGHT  */
    LBRACE = 309,                  /* LBRACE  */
    RBRACE = 310,                  /* RBRACE  */
    LSQBRA = 311,                  /* LSQBRA  */
    RSQBRA = 312,                  /* RSQBRA  */
    DOT = 313,                     /* DOT  */
    COMMA = 314,                   /* COMMA  */
    IF = 315,                      /* IF  */
    ELSE = 316,                    /* ELSE  */
    DO = 317,                      /* DO  */
    WHILE = 318,                   /* WHILE  */
    TR = 319,                      /* TR  */
    FA = 320,                      /* FA  */
    NOT = 321,                     /* NOT  */
    AND = 322,                     /* AND  */
    OR = 323,                      /* OR  */
    NOTEQ = 324,                   /* NOTEQ  */
    EQ = 325,                      /* EQ  */
    GREATER = 326,                 /* GREATER  */
    LESS = 327,                    /* LESS  */
    GREATEREQ = 328,               /* GREATEREQ  */
    LESSEQ = 329,                  /* LESSEQ  */
    PLUS = 330,                    /* PLUS  */
    MINUS = 331,                   /* MINUS  */
    MULTIPLY = 332,                /* MULTIPLY  */
    DIVIDE = 333,                  /* DIVIDE  */
    FFN = 334,                     /* FFN  */
    DATASET = 335,                 /* DATASET  */
    NONLN = 336,                   /* NONLN  */
    SENSEI_OP = 337,               /* SENSEI_OP  */
    INT = 338,                     /* INT  */
    NEW = 339,                     /* NEW  */
    NULL_KEY = 340                 /* NULL_KEY  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 305 "frontend.y"

    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
    ForwardNode* forwardNode;
    TrainingLoopNode* trainingLoopNode;

#line 158 "build/frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED  */
