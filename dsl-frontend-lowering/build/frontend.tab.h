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
    RABBIT_REORDER_OP = 293,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 294,        /* SAMPLE_RANDOM_OP  */
    POW = 295,                     /* POW  */
    COLTILE = 296,                 /* COLTILE  */
    AGGR = 297,                    /* AGGR  */
    FEAT_SIZE_ASSIGN = 298,        /* FEAT_SIZE_ASSIGN  */
    LABEL_SIZE_ASSIGN = 299,       /* LABEL_SIZE_ASSIGN  */
    COARSEN = 300,                 /* COARSEN  */
    INTEGER = 301,                 /* INTEGER  */
    FLOAT = 302,                   /* FLOAT  */
    LBRACE = 303,                  /* LBRACE  */
    RBRACE = 304,                  /* RBRACE  */
    LSQBRA = 305,                  /* LSQBRA  */
    RSQBRA = 306,                  /* RSQBRA  */
    DOT = 307,                     /* DOT  */
    COMMA = 308,                   /* COMMA  */
    IF = 309,                      /* IF  */
    ELSE = 310,                    /* ELSE  */
    DO = 311,                      /* DO  */
    WHILE = 312,                   /* WHILE  */
    TR = 313,                      /* TR  */
    FA = 314,                      /* FA  */
    NOT = 315,                     /* NOT  */
    AND = 316,                     /* AND  */
    OR = 317,                      /* OR  */
    NOTEQ = 318,                   /* NOTEQ  */
    EQ = 319,                      /* EQ  */
    GREATER = 320,                 /* GREATER  */
    LESS = 321,                    /* LESS  */
    GREATEREQ = 322,               /* GREATEREQ  */
    LESSEQ = 323,                  /* LESSEQ  */
    PLUS = 324,                    /* PLUS  */
    MINUS = 325,                   /* MINUS  */
    MULTIPLY = 326,                /* MULTIPLY  */
    DIVIDE = 327,                  /* DIVIDE  */
    FFN = 328,                     /* FFN  */
    DATASET = 329,                 /* DATASET  */
    NONLN = 330,                   /* NONLN  */
    SENSEI_OP = 331,               /* SENSEI_OP  */
    INT = 332,                     /* INT  */
    NEW = 333,                     /* NEW  */
    NULL_KEY = 334                 /* NULL_KEY  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 160 "frontend.y"

    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
    ForwardNode* forwardNode;
    TrainingLoopNode* trainingLoopNode;

#line 152 "build/frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_BUILD_FRONTEND_TAB_H_INCLUDED  */
