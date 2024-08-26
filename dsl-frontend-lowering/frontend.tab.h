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
    RELAXNLN = 274,                /* RELAXNLN  */
    QUANT = 275,                   /* QUANT  */
    GRAPH_ATTR = 276,              /* GRAPH_ATTR  */
    FEAT_ATTR = 277,               /* FEAT_ATTR  */
    RELU = 278,                    /* RELU  */
    RABBIT_REORDER_OP = 279,       /* RABBIT_REORDER_OP  */
    SAMPLE_RANDOM_OP = 280,        /* SAMPLE_RANDOM_OP  */
    COLTILE = 281,                 /* COLTILE  */
    INTEGER = 282,                 /* INTEGER  */
    FLOAT = 283,                   /* FLOAT  */
    LBRACE = 284,                  /* LBRACE  */
    RBRACE = 285,                  /* RBRACE  */
    LSQBRA = 286,                  /* LSQBRA  */
    RSQBRA = 287,                  /* RSQBRA  */
    DOT = 288,                     /* DOT  */
    COMMA = 289,                   /* COMMA  */
    IF = 290,                      /* IF  */
    ELSE = 291,                    /* ELSE  */
    DO = 292,                      /* DO  */
    WHILE = 293,                   /* WHILE  */
    TRUE = 294,                    /* TRUE  */
    FALSE = 295,                   /* FALSE  */
    NOT = 296,                     /* NOT  */
    AND = 297,                     /* AND  */
    OR = 298,                      /* OR  */
    NOTEQ = 299,                   /* NOTEQ  */
    EQ = 300,                      /* EQ  */
    GREATER = 301,                 /* GREATER  */
    LESS = 302,                    /* LESS  */
    GREATEREQ = 303,               /* GREATEREQ  */
    LESSEQ = 304,                  /* LESSEQ  */
    PLUS = 305,                    /* PLUS  */
    MINUS = 306,                   /* MINUS  */
    MULTIPLY = 307,                /* MULTIPLY  */
    DIVIDE = 308,                  /* DIVIDE  */
    FFN = 309,                     /* FFN  */
    DATASET = 310,                 /* DATASET  */
    NONLN = 311,                   /* NONLN  */
    SENSEI_OP = 312,               /* SENSEI_OP  */
    INT = 313,                     /* INT  */
    NEW = 314,                     /* NEW  */
    NULL_KEY = 315                 /* NULL_KEY  */
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

#line 131 "frontend.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_FRONTEND_TAB_H_INCLUDED  */
