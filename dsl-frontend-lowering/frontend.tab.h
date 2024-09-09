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

#ifndef YY_YY_FRONTEND_TAB_H_INCLUDED
# define YY_YY_FRONTEND_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
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
    MODEL = 266,                   /* MODEL  */
    EVAL = 267,                    /* EVAL  */
    TRAIN = 268,                   /* TRAIN  */
    LAYER = 269,                   /* LAYER  */
    LOSS = 270,                    /* LOSS  */
    OPTIMIZER = 271,               /* OPTIMIZER  */
    ITERS = 272,                   /* ITERS  */
    VAL_STEP = 273,                /* VAL_STEP  */
    RMSE_LOSS = 274,               /* RMSE_LOSS  */
    ADAM = 275,                    /* ADAM  */
    RELAXNLN = 276,                /* RELAXNLN  */
    QUANT = 277,                   /* QUANT  */
    GRAPH_ATTR = 278,              /* GRAPH_ATTR  */
    FEAT_ATTR = 279,               /* FEAT_ATTR  */
    RELU = 280,                    /* RELU  */
    RABBIT_REORDER_OP = 281,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 282,        /* SAMPLE_RANDOM_OP  */
    COLTILE = 283,                 /* COLTILE  */
    AGGR = 284,                    /* AGGR  */
    INTEGER = 285,                 /* INTEGER  */
    FLOAT = 286,                   /* FLOAT  */
    LBRACE = 287,                  /* LBRACE  */
    RBRACE = 288,                  /* RBRACE  */
    LSQBRA = 289,                  /* LSQBRA  */
    RSQBRA = 290,                  /* RSQBRA  */
    DOT = 291,                     /* DOT  */
    COMMA = 292,                   /* COMMA  */
    IF = 293,                      /* IF  */
    ELSE = 294,                    /* ELSE  */
    DO = 295,                      /* DO  */
    WHILE = 296,                   /* WHILE  */
    TRUE = 297,                    /* TRUE  */
    FALSE = 298,                   /* FALSE  */
    NOT = 299,                     /* NOT  */
    AND = 300,                     /* AND  */
    OR = 301,                      /* OR  */
    NOTEQ = 302,                   /* NOTEQ  */
    EQ = 303,                      /* EQ  */
    GREATER = 304,                 /* GREATER  */
    LESS = 305,                    /* LESS  */
    GREATEREQ = 306,               /* GREATEREQ  */
    LESSEQ = 307,                  /* LESSEQ  */
    PLUS = 308,                    /* PLUS  */
    MINUS = 309,                   /* MINUS  */
    MULTIPLY = 310,                /* MULTIPLY  */
    DIVIDE = 311,                  /* DIVIDE  */
    FFN = 312,                     /* FFN  */
    DATASET = 313,                 /* DATASET  */
    NONLN = 314,                   /* NONLN  */
    SENSEI_OP = 315,               /* SENSEI_OP  */
    INT = 316,                     /* INT  */
    NEW = 317,                     /* NEW  */
    NULL_KEY = 318                 /* NULL_KEY  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 16 "frontend.y"

    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;

#line 134 "frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_FRONTEND_TAB_H_INCLUDED  */
